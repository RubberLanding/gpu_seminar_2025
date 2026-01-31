import time
import math
import argparse  
import numba
import torch 
import cupy as cp
import numpy as np

from nbody.pytorch_.simulation import compute_forces_pytorch_naive, compute_forces_pytorch_chunked, compute_forces_pytorch_keops, compute_forces_pytorch_matmul, compute_forces_pytorch_optimized
from nbody.cupy_.simulation import compute_forces_cupy_naive, compute_forces_cupy_tiled
from nbody.numba_.simulation import compute_forces_numba_naive, compute_forces_numba_tiled, gpu_step_pos, gpu_step_vel
from nbody.triton_.simulation import compute_forces_triton_naive
from nbody.benchmark.util import print_results, cleanup_gpu

# Constants
G = 6.67430e-11
EPSILON = 1e-4
WARUM_UP_ITER = 5

def measure_time_torch(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_pytorch_naive):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on GPU (PyTorch). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    torch.cuda.empty_cache() 

    pos = torch.tensor(pos_host, device=device, dtype=torch.float32)
    vel = torch.tensor(vel_host, device=device, dtype=torch.float32)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float32)
    N = pos.shape[0]

    dt_tensor = torch.tensor(dt, device=device, dtype=torch.float32)
    dt2_half = 0.5 * dt_tensor * dt_tensor
    dt_half = 0.5 * dt_tensor
    inv_m = 1.0 / mass.unsqueeze(1)

    # Warm-Up 
    with torch.no_grad():
        force_old = compute_forces_func(pos, mass, G, EPSILON).clone()
        for step in range(WARUM_UP_ITER):
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)
            force_new = compute_forces_func(pos, mass, G, EPSILON).clone()
            vel += (force_old + force_new) * inv_m * dt_half
            force_old = force_new
    
    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for step in range(steps):
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)
            force_new = compute_forces_func(pos, mass, G, EPSILON).clone()
            vel += (force_old + force_new) * inv_m * dt_half
            force_old = force_new

    end_event.record()
    torch.cuda.synchronize()
    # Elapsed_time is in ms, convert to seconds
    total_time = start_event.elapsed_time(end_event) / 1000.0
    
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_cupy(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_cupy_naive):
    N = pos_host.shape[0]
    print(f"Running on GPU (CuPy). N={N}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    pos_device = cp.array(pos_host)
    vel_device = cp.array(vel_host)
    mass_device = cp.array(mass_host)

    force_device_old = cp.zeros((N, 3), dtype=cp.float32)
    force_device_new = cp.zeros((N, 3), dtype=cp.float32)

    inv_m = 1.0 / mass_device[:, None]
    dt2_half = 0.5 * dt * dt
    dt_half = 0.5 * dt

    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block

    compute_forces_func((blocks,), (threads_per_block,), (pos_device, mass_device, force_device_old, N, G, EPSILON))

    # Warm-Up
    for step in range(WARUM_UP_ITER):
        pos_device += (vel_device * dt) + (force_device_old * inv_m * dt2_half)
        compute_forces_func((blocks,), (threads_per_block,), (pos_device, mass_device, force_device_new, N, G, EPSILON))
        vel_device += (force_device_old + force_device_new) * inv_m * dt_half
        force_device_old, force_device_new = force_device_new, force_device_old
    
    # Timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()
    for step in range(steps):
        pos_device += (vel_device * dt) + (force_device_old * inv_m * dt2_half)
        compute_forces_func((blocks,), (threads_per_block,), (pos_device, mass_device, force_device_new, N, G, EPSILON))
        vel_device += (force_device_old + force_device_new) * inv_m * dt_half
        force_device_old, force_device_new = force_device_new, force_device_old

    end_event.record()
    end_event.synchronize()
    # Elapsed_time returns ms, convert to seconds
    total_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_numba(pos_host, vel_host, masses_host, dt=0.01, steps=10, compute_forces_func=compute_forces_numba_tiled):
    N = pos_host.shape[0]    
    print(f"Running on GPU (Numba). N={N}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    threads = 128 
    blocks = math.ceil(N / threads)

    d_pos = numba.cuda.to_device(pos_host)
    d_vel = numba.cuda.to_device(vel_host)
    d_mass = numba.cuda.to_device(masses_host)
    d_F_old = numba.cuda.device_array((N, 3), dtype=np.float32)
    d_F_new = numba.cuda.device_array((N, 3), dtype=np.float32)
    
    compute_forces_func[blocks, threads](d_pos, d_mass, d_F_old, G, EPSILON)

    # Warm-Up 
    for step in range(WARUM_UP_ITER):
        gpu_step_pos[blocks, threads](d_pos, d_vel, d_mass, d_F_old, dt)
        compute_forces_func[blocks, threads](d_pos, d_mass, d_F_new, G, EPSILON)
        gpu_step_vel[blocks, threads](d_vel, d_mass, d_F_old, d_F_new, dt)
        d_F_old, d_F_new = d_F_new, d_F_old
    
    # Timing
    start_event = numba.cuda.event()
    end_event = numba.cuda.event()

    start_event.record()
    for step in range(steps):
        gpu_step_pos[blocks, threads](d_pos, d_vel, d_mass, d_F_old, dt)
        compute_forces_func[blocks, threads](d_pos, d_mass, d_F_new, G, EPSILON)
        gpu_step_vel[blocks, threads](d_vel, d_mass, d_F_old, d_F_new, dt)
        d_F_old, d_F_new = d_F_new, d_F_old

    end_event.record()
    end_event.synchronize()
    # Elapsed_time returns ms, convert to seconds
    total_time = numba.cuda.event_elapsed_time(start_event, end_event) / 1000.0

    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N  / total_time
    print_results(total_time, steps_per_second, interactions_per_second, N)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_triton(pos_host, vel_host, mass_host, dt=0.01, steps=10, compute_forces_func=compute_forces_triton_naive):
    device = torch.device("cuda")
    N = pos_host.shape[0]
    pos = torch.from_numpy(pos_host).to(device)
    vel = torch.from_numpy(vel_host).to(device)
    mass = torch.from_numpy(mass_host).to(device)

    dt2_half = 0.5 * dt * dt
    dt_half = 0.5 * dt

    acc_old = compute_forces_func(pos, mass, G, EPSILON)

    # Warm-Up
    for step in range(WARUM_UP_ITER):
        pos += (vel * dt) + (acc_old * dt2_half)
        acc_new = compute_forces_func(pos, mass, G, EPSILON)
        vel += (acc_old + acc_new) * dt_half
        acc_old = acc_new

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for step in range(steps):
        pos += (vel * dt) + (acc_old * dt2_half)
        acc_new = compute_forces_func(pos, mass, G, EPSILON)
        vel += (acc_old + acc_new) * dt_half
        acc_old = acc_new

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
    parser.add_argument("-m", "--method", type=str, choices=["numba", "torch", "cupy", "triton", "all"], default="numba", help="Method to use (numba, torch, cupy, triton or all)")
    parser.add_argument("-f", "--force-func", type=str, choices=["compute_forces_triton_naive", 
                                                                 "compute_forces_cupy_naive", "compute_forces_cupy_tiled",
                                                                 "compute_forces_numba_naive", "compute_forces_numba_tiled",
                                                                 "compute_forces_pytorch_naive", "compute_forces_pytorch_chunked", "compute_forces_pytorch_keops", 
                                                                 "compute_forces_pytorch_matmul", "compute_forces_pytorch_optimized"],
                                                                 help="The force function that is being used, e.g. `gpu_force_kernel_numba_naive` for Numba.")

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
                "compute_forces_numba_tiled": compute_forces_numba_tiled,
            }
        },
        "triton": {
            "measure": measure_time_triton,
            "kernels": {
                "compute_forces_triton_naive": compute_forces_triton_naive,
            }
        },
        "torch": {
            "measure": measure_time_torch,
            "kernels": {
                "compute_forces_pytorch_naive": compute_forces_pytorch_naive,
                "compute_forces_pytorch_chunked": compute_forces_pytorch_chunked,
                "compute_forces_pytorch_keops": compute_forces_pytorch_keops,
                "compute_forces_pytorch_matmul": compute_forces_pytorch_matmul,
                "compute_forces_pytorch_optimized": compute_forces_pytorch_optimized,
            }
        }
    }

    print("START BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")

    # Determine which frameworks to run
    methods_to_run = FRAMEWORK_CONFIG.keys() if args.method == "all" else [args.method]

    for method in methods_to_run:
        config = FRAMEWORK_CONFIG[method]
        force_func = None

        print(f"Measure {method.capitalize()}...")

        # Validation Logic
        if args.force_func:
            if args.force_func in config["kernels"]:
                force_func = config["kernels"][args.force_func]
            else:
                # If a specific force-func was requested but doesn't belong to this method
                print(f"Skipping {method}: '{args.force_func}' is incompatible.")
                print("-" * 20)
                continue
        else:
            print(f"No specific force function provided. Using {method} default.")

        np.random.seed(42) 
        pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
        vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
        mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4

        if force_func is not None: 
            res = config["measure"](pos, vel, mass, dt=args.dt, steps=args.steps, compute_forces_func=force_func)
        else:
            res = config["measure"](pos, vel, mass, dt=args.dt, steps=args.steps)

        # Cleanup GPU memory between different framework runs
        cleanup_gpu()
        if len(methods_to_run) > 1 : print("-" * 20 + "\n")

    print("END BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")