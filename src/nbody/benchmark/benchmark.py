import time
import math
import argparse  
import numba
import torch 
import triton
import cupy as cp
import numpy as np

from nbody.pytorch_.simulation import compute_forces_pytorch_naive, compute_forces_pytorch_chunked, compute_forces_pytorch_keops, compute_forces_pytorch_matmul, compute_forces_pytorch_optimized
from nbody.cupy_.simulation import compute_forces_cupy_naive, compute_forces_cupy_tiled
from nbody.numba_.simulation import compute_forces_numba_naive, compute_forces_numba_tiled, gpu_step_pos, gpu_step_vel
from nbody.triton_.simulation import compute_accel_triton_naive, compute_accel_triton_fast
from nbody.benchmark.util import print_results, cleanup_gpu

# Constants
G = 6.67430e-11
EPSILON = 1e-4
WARUM_UP_ITER = 5

def measure_time_torch(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_pytorch_naive):
    # --- 1. SETUP & TRANSFER ---
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")   
    print(f"Running on GPU (PyTorch). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")
    
    pos  = torch.tensor(pos_host,  device=device, dtype=torch.float32)
    vel  = torch.tensor(vel_host,  device=device, dtype=torch.float32)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float32)
    N    = pos.shape[0]

    # --- 2. CONSTANTS PREP ---
    dt_tensor = torch.tensor(dt, device=device, dtype=torch.float32)
    dt2_half  = 0.5 * dt_tensor * dt_tensor
    dt_half   = 0.5 * dt_tensor
    inv_m     = 1.0 / mass.unsqueeze(1)

    # Warm-Up
    with torch.no_grad():
        # Initial force calculation
        force_old = compute_forces_func(pos, mass, G, EPSILON).clone()
        
        for step in range(WARUM_UP_ITER):
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)
            force_new = compute_forces_func(pos, mass, G, EPSILON).clone()
            vel += (force_old + force_new) * inv_m * dt_half
            force_old = force_new

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        # ==================== CORE LOOP ====================
        for step in range(steps):
            
            # [Step A] Update Position
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)

            # [Step B] Compute Forces
            force_new = compute_forces_func(pos, mass, G, EPSILON).clone()

            # [Step C] Update Velocity
            vel += (force_old + force_new) * inv_m * dt_half

            # [Step D] Swap References
            force_old = force_new
        # ===================================================

    end_event.record()
    torch.cuda.synchronize()
    
    # --- 4. FINALIZE ---
    total_time = start_event.elapsed_time(end_event) / 1000.0
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_cupy(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_cupy_naive):
    # --- 1. SETUP & TRANSFER ---
    print(f"Running on GPU (CuPy). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")
    
    pos  = cp.array(pos_host,  dtype=cp.float32)
    vel  = cp.array(vel_host,  dtype=cp.float32)
    mass = cp.array(mass_host, dtype=cp.float32)
    N    = pos.shape[0]
    
    force_old = cp.zeros((N, 3), dtype=cp.float32)
    force_new = cp.zeros((N, 3), dtype=cp.float32)

    # --- 2. CONSTANTS PREP ---
    dt_vec   = cp.float32(dt)
    dt2_half = 0.5 * dt_vec * dt_vec
    dt_half  = 0.5 * dt_vec
    inv_m    = 1.0 / mass[:, None]

    threads = 128
    blocks  = (N + threads - 1) // threads
    grid_cfg, block_cfg = (blocks,), (threads,)

    # --- 3. WARM-UP ---
    # Note: RawKernel arguments require explicit casting (e.g. np.int32, np.float32)
    compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_old, np.int32(N), np.float32(G), np.float32(EPSILON)))
    
    # Warm-up loop to stabilize GPU clock
    for step in range(WARUM_UP_ITER):
        pos += (vel * dt_vec) + (force_old * inv_m * dt2_half)
        compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_new, np.int32(N), np.float32(G), np.float32(EPSILON)))
        vel += (force_old + force_new) * inv_m * dt_half
        force_old, force_new = force_new, force_old

    # --- 4. START SIMULATION ---
    start_event = cp.cuda.Event()
    end_event   = cp.cuda.Event()

    start_event.record()
    
    # ==================== CORE LOOP ====================
    for step in range(steps):

        # [Step A] Update Position
        pos += (vel * dt_vec) + (force_old * inv_m * dt2_half)

        # [Step B] Compute Forces
        compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_new, np.int32(N), np.float32(G), np.float32(EPS)))

        # [Step C] Update Velocity
        vel += (force_old + force_new) * inv_m * dt_half

        # [Step D] Swap References
        force_old, force_new = force_new, force_old
    # ===================================================

    end_event.record()
    end_event.synchronize()
    
    # --- 5. FINALIZE ---
    total_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_numba(pos_host, vel_host, masses_host, dt=0.01, steps=10, force_func=compute_forces_numba_tiled, threads_per_block=None):
    N = pos_host.shape[0]    
    print(f"Running on GPU (Numba). N={N}, Steps={steps}")
    print(f"Using Force Function: {force_func.__name__}")

    blocks = math.ceil(N / threads_per_block)

    d_pos = numba.cuda.to_device(pos_host)
    d_vel = numba.cuda.to_device(vel_host)
    d_mass = numba.cuda.to_device(masses_host)
    d_F_old = numba.cuda.device_array((N, 3), dtype=np.float32)
    d_F_new = numba.cuda.device_array((N, 3), dtype=np.float32)
    
    force_func[blocks, threads_per_block](d_pos, d_mass, d_F_old, G, EPSILON)

    # Warm-Up 
    for step in range(WARUM_UP_ITER):
        gpu_step_pos[blocks, threads_per_block](d_pos, d_vel, d_mass, d_F_old, dt)
        force_func[blocks, threads_per_block](d_pos, d_mass, d_F_new, G, EPSILON)
        gpu_step_vel[blocks, threads_per_block](d_vel, d_mass, d_F_old, d_F_new, dt)
        d_F_old, d_F_new = d_F_new, d_F_old
    
    # Timing
    start_event = numba.cuda.event()
    end_event = numba.cuda.event()

    start_event.record()
    for step in range(steps):
        gpu_step_pos[blocks, threads_per_block](d_pos, d_vel, d_mass, d_F_old, dt)
        force_func[blocks, threads_per_block](d_pos, d_mass, d_F_new, G, EPSILON)
        gpu_step_vel[blocks, threads_per_block](d_vel, d_mass, d_F_old, d_F_new, dt)
        d_F_old, d_F_new = d_F_new, d_F_old

    end_event.record()
    end_event.synchronize()
    # Elapsed_time returns ms, convert to seconds
    total_time = numba.cuda.event_elapsed_time(start_event, end_event) / 1000.0

    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N  / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_triton(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_accel_triton_naive, block_size=128, ):
    device = torch.device("cuda")
    N = pos_host.shape[0]
    grid = (triton.cdiv(N, block_size),)

    pos = torch.from_numpy(pos_host).to(device)
    vel = torch.from_numpy(vel_host).to(device)
    mass = torch.from_numpy(mass_host).to(device)
    accel_old = torch.empty_like(pos)
    accel_new = torch.empty_like(pos)

    dt2_half = 0.5 * dt * dt
    dt_half = 0.5 * dt

    compute_forces_func[grid](pos, mass, accel_old, G, EPSILON, N, BLOCK_SIZE=block_size)

    # Warm-Up
    for step in range(WARUM_UP_ITER):
        pos += (vel * dt) + (accel_old * dt2_half)
        compute_forces_func[grid](pos, mass, accel_new, G, EPSILON, N, BLOCK_SIZE=block_size)
        vel += (accel_old + accel_new) * dt_half
        accel_old = accel_new

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for step in range(steps):
        pos += (vel * dt) + (accel_old * dt2_half)
        compute_forces_func[grid](pos, mass, accel_new, G, EPS, N, BLOCK_SIZE=block_size)
        vel += (accel_old + accel_new) * dt_half
        accel_old = accel_new

    end_event.record()
    torch.cuda.synchronize()
    # Elapsed_time returns ms, convert to seconds
    total_time = start_event.elapsed_time(end_event) / 1000.0

    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="N-Body Simulation Benchmark")
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of particles")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("-f", "--force-func", type=str, nargs="+", choices=[
            "compute_accel_triton_naive", "compute_accel_triton_fast", "compute_forces_cupy_naive", 
            "compute_forces_cupy_tiled", "compute_forces_numba_naive", 
            "compute_forces_numba_tiled", "compute_forces_pytorch_naive", 
            "compute_forces_pytorch_chunked", "compute_forces_pytorch_keops", 
            "compute_forces_pytorch_matmul", "compute_forces_pytorch_optimized"], 
            help="One or more force functions to benchmark.")
    parser.add_argument("--store-results", action="store_true", help="Store the results.")
    parser.add_argument("--store-plot", action="store_true", help="Store the performance plot.") 
    parser.add_argument("--tpb-numba", type=int, default=128, help="Threads per block for Numba. Should be a multiple of 32.")

    args = parser.parse_args()

    # Mapping of framework name to its measure function and allowed force kernels
    FRAMEWORK_CONFIG = {
        "cupy": {
            "measure": measure_time_cupy,
            "kernels": {
                "compute_forces_cupy_naive": compute_forces_cupy_naive,
                "compute_forces_cupy_tiled": compute_forces_cupy_tiled,
            }
        },
        "numba": {
            "measure": measure_time_numba,
            "kernels": {
                "compute_forces_numba_naive": compute_forces_numba_naive,
                "compute_forces_numba_tiled": compute_forces_numba_tiled(args.tpb_numba),
            }
        },
        "triton": {
            "measure": measure_time_triton,
            "kernels": {
                "compute_accel_triton_naive": compute_accel_triton_naive,
                "compute_accel_triton_fast": compute_accel_triton_fast,
            }
        },
        "pytorch": {
            "measure": measure_time_torch,
            "kernels": {
                "compute_forces_pytorch_naive":     compute_forces_pytorch_naive,
                "compute_forces_pytorch_chunked":   compute_forces_pytorch_chunked,
                "compute_forces_pytorch_keops":     compute_forces_pytorch_keops,
                "compute_forces_pytorch_matmul":    compute_forces_pytorch_matmul,
                "compute_forces_pytorch_optimized": compute_forces_pytorch_optimized,
                }
        }
    }

    print("START BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")

    for force_func_str in args.force_func: 
        if "numba" in force_func_str:
            framework = "numba"
        elif "cupy" in force_func_str:
            framework = "cupy"
        elif "pytorch" in force_func_str:
            framework = "pytorch"
        elif "triton" in force_func_str:
            framework = "triton"

        config = FRAMEWORK_CONFIG[framework]
        print(f"Measure {force_func_str.capitalize()}...")

        force_func = config["kernels"][force_func_str]
        np.random.seed(42) 
        pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
        vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
        mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4

        res = config["measure"](pos, vel, mass, dt=args.dt, steps=args.steps, compute_forces_func=force_func)

        # Cleanup GPU memory between different framework runs
        cleanup_gpu()
        if len(args.force_func) > 1 : print("-" * 20 + "\n")

    print("END BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")