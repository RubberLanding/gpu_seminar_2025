import time
import math
import argparse  # <--- Added for CLI args
import numba
import torch 
import cupy as cp
import numpy as np

from nbody.pytorch_.simulation import compute_forces_pytorch
from nbody.cupy_.simulation import compute_forces_cupy
from nbody.numba_.simulation import gpu_force_kernel_numba, gpu_step_pos, gpu_step_vel
from nbody.numba_.simulation import cpu_force_kernel_numba, cpu_step_pos, cpu_step_vel

# Constants
G = 6.67430e-11
EPSILON = 1e-4

def measure_time_torch(pos_host, vel_host, mass_host, dt, steps):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} (PyTorch). N={pos_host.shape[0]}, Steps={steps}")

    # Empty cache before running big probem instances with lots of allocations
    torch.cuda.empty_cache() 

    # Move data to GPU
    pos = torch.tensor(pos_host, device=device, dtype=torch.float64)
    vel = torch.tensor(vel_host, device=device, dtype=torch.float64)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float64)

    N = pos.shape[0]

    # Pre-calculate constants
    dt_tensor = torch.tensor(dt, device=device, dtype=torch.float64)
    dt2_half = 0.5 * dt_tensor * dt_tensor
    dt_half = 0.5 * dt_tensor
    inv_m = 1.0 / mass.unsqueeze(1)
    
    # Sync before timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.perf_counter()

    with torch.no_grad():
        # Initial Force
        force_old = compute_forces_pytorch(pos, mass, G, EPSILON)

        for step in range(steps):
            # Update position
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)
            # Upate forces
            force_new = compute_forces_pytorch(pos, mass, G, EPSILON)
            # Update velocity
            vel += (force_old + force_new) * inv_m * dt_half
            # Swap references
            force_old = force_new

    # Sync before timing
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.perf_counter()

    total_time = end_time - start_time
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print("-" * 30)
    print(f"Total Runtime:            {total_time:.4f} seconds")
    print(f"Performance Steps:        {steps_per_second:.2f} steps/second")
    print(f"Performance Interactions: {interactions_per_second:.2f} interactions/second")
    print("-" * 30, "\n")
    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_cupy(pos_host, vel_host, mass_host, dt, steps):
    N = pos_host.shape[0]
    print(f"Running on GPU (CuPy). N={N}, Steps={steps}")

    # Move data to GPU using CuPy 
    pos_device = cp.array(pos_host)
    vel_device = cp.array(vel_host)
    mass_device = cp.array(mass_host)

    # Allocate force buffers on GPU
    force_device_old = cp.zeros((N, 3), dtype=cp.float64)
    force_device_new = cp.zeros((N, 3), dtype=cp.float64)

    # Pre-calculate constants
    inv_m = 1.0 / mass_device[:, None]
    dt2_half = 0.5 * dt * dt
    dt_half = 0.5 * dt

    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block

    # Sync before start
    cp.cuda.Stream.null.synchronize()
    start_time = time.perf_counter()

    # Initial force calculation
    compute_forces_cupy((blocks,), (threads_per_block,), (pos_device, mass_device, force_device_old, N, G, EPSILON))

    for step in range(steps):
        pos_device += (vel_device * dt) + (force_device_old * inv_m * dt2_half)
        compute_forces_cupy((blocks,), (threads_per_block,), (pos_device, mass_device, force_device_new, N, G, EPSILON))
        vel_device += (force_device_old + force_device_new) * inv_m * dt_half
        force_device_old, force_device_new = force_device_new, force_device_old

    # Sync before end
    cp.cuda.Stream.null.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    steps_per_second = steps / total_time
    interactions_per_second = steps * N * N / total_time
    print("-" * 30)
    print(f"Total Runtime:            {total_time:.4f} seconds")
    print(f"Performance Steps:        {steps_per_second:.2f} steps/second")
    print(f"Performance Interactions: {interactions_per_second:.2f} interactions/second")
    print("-" * 30, "\n")
    return steps, total_time, steps_per_second, interactions_per_second

def measure_time_numba(pos_host, vel_host, masses_host, dt, steps, device="auto"):
    N = pos_host.shape[0]
    use_gpu = (device == "gpu" or device == "auto") and numba.cuda.is_available()
    
    if use_gpu:
        print(f"Running on GPU (Numba). N={N}, Steps={steps}")
        threads = 256
        blocks = math.ceil(N / threads)

        d_pos = numba.cuda.to_device(pos_host)
        d_vel = numba.cuda.to_device(vel_host)
        d_mass = numba.cuda.to_device(masses_host)
        d_F_old = numba.cuda.device_array((N, 3), dtype=np.float64)
        d_F_new = numba.cuda.device_array((N, 3), dtype=np.float64)

        # Sync before start
        numba.cuda.synchronize()
        start_time = time.perf_counter()
        
        gpu_force_kernel_numba[blocks, threads](d_pos, d_mass, d_F_old)
        
        for step in range(steps):
            gpu_step_pos[blocks, threads](d_pos, d_vel, d_mass, d_F_old, dt)
            gpu_force_kernel_numba[blocks, threads](d_pos, d_mass, d_F_new)
            gpu_step_vel[blocks, threads](d_vel, d_mass, d_F_old, d_F_new, dt)
            d_F_old, d_F_new = d_F_new, d_F_old
        
        # Sync before end
        numba.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        steps_per_second = steps / total_time
        interactions_per_second = steps * N * N  / total_time
        print("-" * 30)
        print(f"Total Runtime:            {total_time:.4f} seconds")
        print(f"Performance Steps:        {steps_per_second:.2f} steps/second")
        print(f"Performance Interactions: {interactions_per_second:.2f} interactions/second")
        print("-" * 30, "\n")
        return steps, total_time, steps_per_second, interactions_per_second

    else:
        print(f"Running on CPU (Numba). N={N}, Steps={steps}")
        r_pos = pos_host.copy()
        v_vel = vel_host.copy()
        masses = masses_host
        F_old = np.zeros_like(r_pos)
        F_new = np.zeros_like(r_pos)

        start_time = time.perf_counter()
        cpu_force_kernel_numba(r_pos, masses, F_old)
        for step in range(steps):
            cpu_step_pos(r_pos, v_vel, masses, F_old, dt)  
            cpu_force_kernel_numba(r_pos, masses, F_new)
            cpu_step_vel(v_vel, masses, F_old, F_new, dt)
            F_old[:] = F_new[:] 
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        steps_per_second = steps / total_time
        interactions_per_second = steps * N * N / total_time
        print("-" * 30)
        print(f"Total Runtime:            {total_time:.4f} seconds")
        print(f"Performance Steps:        {steps_per_second:.2f} steps/second")
        print(f"Performance Interactions: {interactions_per_second:.2f} interactions/second")
        print("-" * 30, "\n")
        return steps, total_time, steps_per_second, interactions_per_second


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="N-Body Simulation Benchmark")
    
    # Define arguments
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of bodies")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("-m", "--method", type=str, choices=["numba", "torch", "cupy", "all"], default="numba", help="Method to use (numba, torch, cupy, or all)")
    
    args = parser.parse_args()

    # Generate Initial Data
    print(f"Initializing {args.num_bodies} bodies...")
    pos = np.random.rand(args.num_bodies, 3).astype(np.float64) * 100.0
    vel = np.random.rand(args.num_bodies, 3).astype(np.float64) - 0.5
    mass = np.random.rand(args.num_bodies).astype(np.float64) * 1e4
    
    # Run Benchmark
    if args.method == "numba" or args.method == "all":
        measure_time_numba(pos, vel, mass, args.dt, args.steps)
        
    if args.method == "torch" or args.method == "all":
        measure_time_torch(pos, vel, mass, args.dt, args.steps)
        
    if args.method == "cupy" or args.method == "all":
        measure_time_cupy(pos, vel, mass, args.dt, args.steps)