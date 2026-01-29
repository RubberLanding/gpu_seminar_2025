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

def measure_time_torch(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_pytorch_naive):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on GPU (PyTorch). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    # Empty cache before running big probem instances with lots of allocations
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
    
    # Sync before timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.perf_counter()

    # The compuation that is being measured
    with torch.no_grad():
        for step in range(steps):
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)
            force_new = compute_forces_func(pos, mass, G, EPSILON).clone()
            vel += (force_old + force_new) * inv_m * dt_half
            force_old = force_new

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.perf_counter()

    total_time = end_time - start_time
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second)

    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_cupy(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_cupy_naive):
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
    
    # Sync before start
    cp.cuda.Device().synchronize()
    start_time = time.perf_counter()

    # The compuation that is being measured
    for step in range(steps):
        pos_device += (vel_device * dt) + (force_device_old * inv_m * dt2_half)
        compute_forces_func((blocks,), (threads_per_block,), (pos_device, mass_device, force_device_new, N, G, EPSILON))
        vel_device += (force_device_old + force_device_new) * inv_m * dt_half
        force_device_old, force_device_new = force_device_new, force_device_old

    # Sync before end
    cp.cuda.Device().synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second)
    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_numba(pos_host, vel_host, masses_host, dt, steps, compute_forces_func=compute_forces_numba_tiled):
    N = pos_host.shape[0]    
    print(f"Running on GPU (Numba). N={N}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    threads = 128 # When using `gpu_force_kernel_numba_tiled`, this must be the same value as in TPB in `simulation.py`
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
    
    # Sync before start
    numba.cuda.synchronize()
    start_time = time.perf_counter()
    
    # Actual computation that is being measured
    for step in range(steps):
        gpu_step_pos[blocks, threads](d_pos, d_vel, d_mass, d_F_old, dt)
        compute_forces_func[blocks, threads](d_pos, d_mass, d_F_new, G, EPSILON)
        gpu_step_vel[blocks, threads](d_vel, d_mass, d_F_old, d_F_new, dt)
        d_F_old, d_F_new = d_F_new, d_F_old
    
    # Sync before end
    numba.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N  / total_time
    print_results(total_time, steps_per_second, interactions_per_second)
    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_triton(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_triton_naive):
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

    # Sync before timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.perf_counter()

    for step in range(steps):
        pos += (vel * dt) + (acc_old * dt2_half)
        acc_new = compute_forces_func(pos, mass, G, EPSILON)
        vel += (acc_old + acc_new) * dt_half
        acc_old = acc_new

    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.perf_counter()

    total_time = end_time - start_time
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print_results(total_time, steps_per_second, interactions_per_second)

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

    print(f"Measure {args.method}.")
    print(f"Initializing {args.num_bodies} bodies...")
    pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
    vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
    mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4
    
    if args.method == "cupy" or args.method == "all":
        if args.force_func=="compute_forces_cupy_naive": force_func = compute_forces_cupy_naive 
        elif args.force_func=="compute_forces_cupy_tiled": force_func = compute_forces_cupy_tiled
        else: print(f"The method to compute the forces {args.force_func} does not match the computation method {args.method}.")

        if args.force_func:
            results = measure_time_cupy(pos, vel, mass, args.dt, args.steps, compute_forces_func=force_func)
        else:
            results = measure_time_cupy(pos, vel, mass, args.dt, args.steps)
        cleanup_gpu()

    if args.method == "numba" or args.method == "all":
        if args.force_func=="compute_forces_numba_naive": force_func = compute_forces_numba_naive
        elif args.force_func=="compute_forces_numba_tiled": force_func = compute_forces_numba_tiled
        else: print(f"The method to compute the forces {args.force_func} does not match the computation method {args.method}.")

        if args.force_func:
            results = measure_time_numba(pos, vel, mass, args.dt, args.steps, compute_forces_func=force_func)
        else:
            results = measure_time_numba(pos, vel, mass, args.dt, args.steps)  
        cleanup_gpu()

    if args.method == "triton" or args.method == "all":
        if args.force_func=="compute_forces_triton_naive": force_func = compute_forces_triton_naive
        else: print(f"The method to compute the forces {args.force_func} does not match the computation method {args.method}.")

        if args.force_func:
            results = measure_time_triton(pos, vel, mass, args.dt, args.steps, compute_forces_func=force_func)
        else:
            results = measure_time_triton(pos, vel, mass, args.dt, args.steps)
        cleanup_gpu()

    if args.method == "torch" or args.method == "all":
        if args.force_func=="compute_forces_pytorch_naive": force_func = compute_forces_pytorch_naive 
        elif args.force_func=="compute_forces_pytorch_chunked": force_func = compute_forces_pytorch_chunked  
        elif args.force_func=="compute_forces_pytorch_keops": force_func = compute_forces_pytorch_keops
        elif args.force_func=="compute_forces_pytorch_optimized": force_func = compute_forces_pytorch_optimized
        elif args.force_func=="compute_forces_pytorch_matmul": force_func = compute_forces_pytorch_matmul
        else: print(f"The method to compute the forces {args.force_func} does not match the computation method {args.method}.")

        if args.force_func:
            results = measure_time_torch(pos, vel, mass, args.dt, args.steps, compute_forces_func=force_func)
        else:
            results = measure_time_torch(pos, vel, mass, args.dt, args.steps)
        cleanup_gpu()
