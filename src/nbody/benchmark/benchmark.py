import math
import argparse  
import numba
import torch 
import triton
import cupy as cp
import numpy as np

from nbody.pytorch_.simulation import compute_forces_pytorch_naive, compute_forces_pytorch_chunked, compute_forces_pytorch_keops, compute_forces_pytorch_matmul, compute_forces_pytorch_optimized, set_triton_config
from nbody.cupy_.simulation import compute_forces_cupy_naive, compute_forces_cupy_tiled, compute_forces_cupy_keops
from nbody.numba_.simulation import compute_forces_numba_naive, compute_forces_numba_tiled, gpu_step_pos, gpu_step_vel
from nbody.triton_.simulation import compute_accel_triton_naive, compute_accel_triton_tensor, compute_accel_triton_tiled, compute_accel_triton_mixed
from nbody.benchmark.util import print_results, cleanup_gpu

# Constants
G = 6.67430e-11
EPSILON = 1e-4
WARUM_UP_ITER = 5

def measure_time_torch(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_pytorch_naive, triton_block_size=4096):
    # --- SETUP & TRANSFER ---
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")   
    print(f"Running on GPU (PyTorch). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    # cleanup_gpu()
    # set_triton_config(triton_block_size)
    
    pos  = torch.tensor(pos_host,  device=device, dtype=torch.float32)
    vel  = torch.tensor(vel_host,  device=device, dtype=torch.float32)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float32)
    N    = pos.shape[0]

    # --- CONSTANTS PREP ---
    dt_tensor = torch.tensor(dt, device=device, dtype=torch.float32)
    dt2_half  = 0.5 * dt_tensor * dt_tensor
    dt_half   = 0.5 * dt_tensor
    inv_m     = 1.0 / mass.unsqueeze(1)

    # --- WARM-UP ---
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

    # --- START SIMULATION ---
    with torch.no_grad():
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
    
    # --- FINALIZE ---
    total_time = start_event.elapsed_time(end_event) / 1000.0
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_cupy(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_cupy_naive, threads=128):
    # --- SETUP & TRANSFER ---
    print(f"Running on GPU (CuPy). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")
    
    pos  = cp.array(pos_host,  dtype=cp.float32)
    vel  = cp.array(vel_host,  dtype=cp.float32)
    mass = cp.array(mass_host, dtype=cp.float32)
    N    = pos.shape[0]
    
    force_old = cp.zeros((N, 3), dtype=cp.float32)
    force_new = cp.zeros((N, 3), dtype=cp.float32)

    # --- CONSTANTS PREP ---
    dt_vec   = cp.float32(dt)
    dt2_half = 0.5 * dt_vec * dt_vec
    dt_half  = 0.5 * dt_vec
    inv_m    = 1.0 / mass[:, None]

    blocks  = (N + threads - 1) // threads
    grid_cfg, block_cfg = (blocks,), (threads,)

    # --- WARM-UP ---
    # Initial force calculation
    compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_old, np.int32(N), np.float32(G), np.float32(EPSILON)))
    
    # Warm-up loop
    for step in range(WARUM_UP_ITER):
        pos += (vel * dt_vec) + (force_old * inv_m * dt2_half)
        compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_new, np.int32(N), np.float32(G), np.float32(EPSILON)))
        vel += (force_old + force_new) * inv_m * dt_half
        force_old, force_new = force_new, force_old

    start_event = cp.cuda.Event()
    end_event   = cp.cuda.Event()
    start_event.record()

    # --- START SIMULATION ---
    for step in range(steps):

        # [Step A] Update Position
        pos += (vel * dt_vec) + (force_old * inv_m * dt2_half)

        # [Step B] Compute Forces
        compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_new, np.int32(N), np.float32(G), np.float32(EPSILON)))

        # [Step C] Update Velocity
        vel += (force_old + force_new) * inv_m * dt_half

        # [Step D] Swap References
        force_old, force_new = force_new, force_old
    # ===================================================

    end_event.record()
    end_event.synchronize()
    
    # --- FINALIZE ---
    total_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_numba(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_numba_naive, threads=128):
    # --- SETUP & TRANSFER ---
    N = pos_host.shape[0]
    print(f"Running on GPU (Numba). N={N}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    pos  = numba.cuda.to_device(pos_host)
    vel  = numba.cuda.to_device(vel_host)
    mass = numba.cuda.to_device(mass_host)
    force_old = numba.cuda.device_array((N, 3), dtype=np.float32)
    force_new = numba.cuda.device_array((N, 3), dtype=np.float32)

    # --- CONSTANTS PREP ---
    blocks  = math.ceil(N / threads)

    if compute_forces_func == compute_forces_numba_tiled: 
        compute_forces_func = compute_forces_func(threads)

    # --- WARM-UP ---
    # Initial force calculation
    compute_forces_func[blocks, threads](pos, mass, force_old, G, EPSILON)
    for step in range(WARUM_UP_ITER):
        gpu_step_pos[blocks, threads](pos, vel, mass, force_old, dt)
        compute_forces_func[blocks, threads](pos, mass, force_new, G, EPSILON)
        gpu_step_vel[blocks, threads](vel, mass, force_old, force_new, dt)
        force_old, force_new = force_new, force_old

    start_event = numba.cuda.event()
    end_event   = numba.cuda.event()
    start_event.record()

    # --- START SIMULATION ---
    for step in range(steps):

        # [Step A] Update Position: r(t+dt) = r(t) + v(t)dt + 0.5*a(t)dt^2
        gpu_step_pos[blocks, threads](pos, vel, mass, force_old, dt)

        # [Step B] Compute Forces: F(t+dt)
        compute_forces_func[blocks, threads](pos, mass, force_new, G, EPSILON)

        # [Step C] Update Velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))dt
        gpu_step_vel[blocks, threads](vel, mass, force_old, force_new, dt)

        # [Step D] Swap References
        force_old, force_new = force_new, force_old
    # ===================================================

    end_event.record()
    end_event.synchronize()
    
    # --- FINALIZE ---
    total_time = numba.cuda.event_elapsed_time(start_event, end_event) / 1000.0
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_triton(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_accel_triton_tiled, block_size=32):
    # --- SETUP & TRANSFER ---
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")   
    N = pos_host.shape[0]
    print(f"Running on GPU (Triton). N={N}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    pos  = torch.tensor(pos_host,  device=device, dtype=torch.float32)
    vel  = torch.tensor(vel_host,  device=device, dtype=torch.float32)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float32)
    force_old = torch.empty_like(pos)
    force_new = torch.empty_like(pos)

    use_mixed = "mixed" in compute_forces_func.__name__
    mass_16 = mass.to(torch.float16) if use_mixed else mass

    # --- CONSTANTS PREP ---
    dt_vec   = torch.tensor(dt, device=device, dtype=torch.float32)
    dt2_half = 0.5 * dt_vec * dt_vec
    dt_half  = 0.5 * dt_vec
    
    grid = (triton.cdiv(N, block_size),)

    # --- WARM-UP ---
    # WARNING: Although named 'force_old/new' and 'compute_forces_func' to match other backends,
    # the Triton kernel calculates ACCELERATION directly (Force / Mass).
    # Therefore, we DO NOT multiply by inv_m in the update steps below.

    # Initial force calculation 
    if use_mixed:
        compute_forces_func[grid](pos.to(torch.float16), mass_16, force_old, G, EPSILON, N, BLOCK_SIZE=block_size)
    else:
        compute_forces_func[grid](pos, mass, force_old, G, EPSILON, N, BLOCK_SIZE=block_size)

    for step in range(WARUM_UP_ITER):
        pos += (vel * dt_vec) + (force_old * dt2_half) 
        compute_forces_func[grid](pos, mass, force_new, G, EPSILON, N, BLOCK_SIZE=block_size)
        vel += (force_old + force_new) * dt_half
        force_old, force_new = force_new, force_old

    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    
    start_event.record()

    # --- START SIMULATION ---
    for step in range(steps):

        # [Step A] Update Position: r(t+dt) = r(t) + v(t)dt + 0.5*a(t)dt^2
        pos += (vel * dt_vec) + (force_old * dt2_half)

        # [Step B] Compute Acceleration: a(t+dt)
        if use_mixed:
            compute_forces_func[grid](pos.to(torch.float16), mass_16, force_new, G, EPSILON, N, BLOCK_SIZE=block_size)
        else:
            compute_forces_func[grid](pos, mass, force_new, G, EPSILON, N, BLOCK_SIZE=block_size)

        # [Step C] Update Velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))dt
        vel += (force_old + force_new) * dt_half

        # [Step D] Swap References
        force_old, force_new = force_new, force_old
    # ===================================================

    end_event.record()
    torch.cuda.synchronize()
    
    # --- FINALIZE ---
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
            "compute_accel_triton_naive", "compute_accel_triton_tensor", "compute_accel_triton_tiled", "compute_accel_triton_mixed",
            "compute_forces_cupy_naive", "compute_forces_cupy_tiled", "compute_forces_cupy_keops",
            "compute_forces_numba_naive", "compute_forces_numba_tiled", 
            "compute_forces_pytorch_naive", "compute_forces_pytorch_chunked", "compute_forces_pytorch_keops", 
            "compute_forces_pytorch_matmul", "compute_forces_pytorch_optimized"], 
            help="One or more force functions to benchmark.")
    parser.add_argument("-sr", "--store-results", action="store_true", help="Store the results.")
    parser.add_argument("-sp", "--store-plot", action="store_true", help="Store the performance plot.") 
    parser.add_argument("-tn", "--tpb-numba", type=int, default=128, help="Threads per block for Numba. Should be a multiple of 32.")
    parser.add_argument("-bt", "--bs-triton", type=int, default=32, help="Block size for Triton. Should be a multiple of 16.")
    args = parser.parse_args()

    assert args.force_func != None, "Provide a force function, e.g. `--force-func compute_forces_cupy_naive`!"

    # Mapping of framework name to its measure function and allowed force kernels
    FRAMEWORK_CONFIG = {
        "cupy": {
            "measure": measure_time_cupy,
            "kernels": {
                "compute_forces_cupy_naive": compute_forces_cupy_naive,
                "compute_forces_cupy_tiled": compute_forces_cupy_tiled,
                "compute_forces_cupy_keops": compute_forces_cupy_keops,
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
                "compute_accel_triton_tensor": compute_accel_triton_tensor,
                "compute_accel_triton_tiled": compute_accel_triton_tiled,
                "compute_accel_triton_mixed": compute_accel_triton_mixed,
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

        # Framework-specific arguments
        measure_kwargs = {}
        if framework == "triton":
            measure_kwargs["block_size"] = args.bs_triton

        np.random.seed(42) 
        pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
        vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
        mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4

        res = config["measure"](pos, vel, mass, dt=args.dt, steps=args.steps, compute_forces_func=force_func, **measure_kwargs)

        # Cleanup GPU memory between different framework runs
        cleanup_gpu()
        if len(args.force_func) > 1 : print("-" * 20 + "\n")

    print("END BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")