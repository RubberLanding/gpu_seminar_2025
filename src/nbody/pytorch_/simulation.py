"""
N-Body Simulation in PyTorch
=======================================
This module compares four approaches for calculating pairwise gravitational interactions using Pytorch. 

1. Naive (Eager) Vectorization: `compute_forces_pytorch_naive_()`
------------------------------
The standard PyTorch 'Eager' approach using broadcasting to create an (N, N, 3) 
displacement tensor. Only feasible for N ~ 60000, since the intermediate tensors like `diff` lead 
to OOM. Every operation (+, -, *, pow) launches a separate CUDA kernel, forcing multiple round-trips to DRAM.


2. Naive Compiled (torch.compile): `compute_forces_pytorch_naive()`
---------------------------------
Wraps the naive vectorized logic in `torch.compile()` to compile with Triton: `compiled_fn = torch.compile(naive_fn)`.
Triton attempts tiling the workload. For N ~ 8000 this triggers `TRITON_MAX_BLOCK` errors. These can be (somewhat) circumvented 
by increasing the max block size (see below).
 

3. Chunked (Tiled) Approach: `compute_forces_pytorch_chunked_()`
---------------------------
Manually tiles the computation by iterating over chunks of the target particles 
while broadcasting against all source particles. Can be compiled using Triton, available with `compute_forces_pytorch_chunked()`, while
same limitations concerning the block size apply.


4. PyKeOps (Symbolic Tensors): `compute_forces_pytorch_keops()`
-----------------------------
Uses the KeOps library to define a symbolic 'LazyTensor' reduction. Requires `nvcc` to be available. 
Only approach that has O(N) memory complexity (instead of N^2) by performing the reduction in registers/SRAM, 
bypassing DRAM bottlenecks almost entirely.
"""

import torch
import numpy as np
import argparse
from torch.cuda import nvtx

# Keep the precision of float32 for the range but use the speed of float16 for the math
torch.set_float32_matmul_precision('high')

# Change Tritons block size depending on the size of the simulation 
def set_triton_config(block_size):
    print(f"Setting TRITON_MAX_BLOCK['X'] to {block_size}...")
    
    try:
        from torch._inductor.runtime import triton_heuristics
        triton_heuristics.TRITON_MAX_BLOCK["X"] = block_size
    except (ImportError, AttributeError, KeyError):
        print("Warning: Could not patch triton_heuristics")
    try:
        from torch._inductor.runtime import hints
        hints.TRITON_MAX_BLOCK["X"] = block_size
    except (ImportError, AttributeError, KeyError):
        print("Warning: Could not patch hints")
    # try:
    #     import torch._inductor.config as config
    #     # This forces the compiler to use cuBLAS instead of generating Triton kernels for matmuls
    #     config.freezing = True
    #     config.triton.desugared_library_calls = True
    #     # This is the "Magic" flag that stops Inductor from trying to out-think cuBLAS
    #     config.coordinate_descent_tuning = True
    # except (ImportError, AttributeError, KeyError):
    #     print("Warning: Could not patch config")


# Constants
G = 6.67430e-11
EPSILON = 1e-4

@torch.compile(mode="max-autotune")
# Separate the calculation into smaller chunks to fit into RAM
def compute_forces_pytorch_chunked(pos, mass, G, EPSILON, chunk_size=128):
    """
    Optimized chunked force calculation.
    Reduces peak VRAM usage by avoiding the explicit (Chunk, N, 3) acceleration tensor.
    """
    N = pos.shape[0]
    forces = torch.empty_like(pos)
    
    # Pre-unsqueeze mass to (1, N) for broadcasting
    mass_j = mass.unsqueeze(0) 

    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        
        # Target particles (Chunk, 1, 3)
        pos_chunk = pos[i:end_i].unsqueeze(1)
        
        # Source particles (1, N, 3)
        pos_all = pos.unsqueeze(0)

        # Compute Diff (Chunk, N, 3)
        diff = pos_all - pos_chunk  
        
        # Compute Distances
        dist_sq = diff.pow(2).sum(dim=-1) + EPSILON**2
        
        # Compute Scalar Weight (Chunk, N)
        # weight = m_j / (r^2 + eps^2)^1.5
        scalar_weight = mass_j * dist_sq.pow(-1.5)
        
        # Compute Weighted Sum directly
        # We broadcast (Chunk, N, 1) * (Chunk, N, 3) and sum over N
        # This re-uses the 'diff' memory tensor or registers without allocating a new big block
        accel_chunk = (diff * scalar_weight.unsqueeze(-1)).sum(dim=1)
        
        # Apply constants
        forces[i:end_i] = G * mass[i:end_i].unsqueeze(1) * accel_chunk
        
    return forces

# IMPORTANT: nvcc must be available on the system before running this, e.g. by loading CUDA
# Using PyKeOps to avoid allocating N^2 memory
from pykeops.torch import LazyTensor
def compute_forces_pytorch_keops(pos, mass, G, EPSILON):
    # x_i: target particles (N, 1, 3)
    x_i = LazyTensor(pos[:, None, :])
    # y_j: source particles (1, N, 3)
    y_j = LazyTensor(pos[None, :, :])
    # m_j: source masses (1, N, 1)
    m_j = LazyTensor(mass[None, :, None])

    # Symbolic computation (no memory allocated yet)
    diff = x_i - y_j
    sq_dist = (diff ** 2).sum(-1)
    inv_dist_cube = (sq_dist + EPSILON**2).rsqrt() ** 3
    
    # The reduction happens here automatically in a fused CUDA kernel
    ftmp = (diff * m_j * inv_dist_cube).sum(1)

    return -G * mass.unsqueeze(1) * ftmp

@torch.compile(mode="max-autotune")
def compute_forces_pytorch_matmul(pos, mass, G, EPSILON):
    N = pos.shape[0]
    
    # 1. Squared Distances - This is our ONE allowed giant buffer (23.8 GiB)
    dot_pos = torch.sum(pos**2, dim=1, keepdim=True)
    # We use aten.mm but immediately start reusing the result
    res = torch.ops.aten.mm(pos, pos.t()) 
    
    # In-place transform 'res' into 'dist_sq'
    # dist_sq = dot_pos + dot_pos.t() - 2.0 * res
    res.mul_(-2.0).add_(dot_pos).add_(dot_pos.t())
    res.clamp_(min=0) # Now 'res' is dist_sq
    
    # 2. In-place transform 'dist_sq' into 'inv_dist_cube'
    res.add_(EPSILON**2).pow_(-1.5).mul_(G)
    
    # 3. Mask the diagonal WITHOUT creating a new matrix
    res.fill_diagonal_(0.0)
    
    # 4. Apply SOURCE masses (m_j) in-place
    # 'res' now becomes the 'weights' matrix
    res.mul_(mass.view(1, -1))

    # 5. Final vector math
    # Term 1: uses the weights matrix (res)
    term1 = torch.mm(res, pos) 
    # Term 2: reduction happens on the same buffer
    term2 = pos * res.sum(dim=1, keepdim=True)
    
    return mass.view(-1, 1) * (term1 - term2)

@torch.compile(mode="max-autotune")
def compute_forces_pytorch_optimized(pos, mass, G, EPSILON):
    dot_pos = torch.sum(pos**2, dim=1, keepdim=True)
    # This is the "Heavy Lifting" - GEMM is 10x faster than broadcasting subtraction
    dist_sq = dot_pos + dot_pos.t() - 2.0 * torch.mm(pos, pos.t())
    
    inv_dist_cube = (dist_sq.clamp(min=0) + EPSILON**2).pow(-1.5)
    
    # Functional mask to avoid the OOM from torch.eye
    # We only zero the diagonal
    weights = (inv_dist_cube * mass[None, :]) * G
    mask = torch.eye(weights.shape[0], device=weights.device).logical_not()
    weights = weights * mask

    term1 = torch.mm(weights, pos)
    term2 = pos * weights.sum(dim=1, keepdim=True)
    
    return mass[:, None] * (term1 - term2)

# Regular approach
# Chunk size is a dummy argument and can be removed in later versions
@torch.compile(mode="max-autotune")
def compute_forces_pytorch_naive(pos, mass, G, EPSILON):
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)

    dist_sq = (diff ** 2).sum(dim=-1)
    dist = (dist_sq + EPSILON**2).sqrt()

    inv_dist_cube = dist.pow(-3)
    mass_j = mass.unsqueeze(0)
    
    scalar_factor = (mass_j * inv_dist_cube).unsqueeze(-1)
    ftmp = (diff * scalar_factor).sum(dim=1)
    force = -G * mass.unsqueeze(1) * ftmp
    
    return force

def run_simulation_torch(pos_host, vel_host, mass_host, dt, steps, compute_force_func=compute_forces_pytorch_naive, store_history=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} (PyTorch). N={pos_host.shape[0]}, Steps={steps}")

    # Move data to GPU
    pos = torch.tensor(pos_host, device=device, dtype=torch.float32)
    vel = torch.tensor(vel_host, device=device, dtype=torch.float32)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float32)
    N = pos.shape[0]

    # Allocate history buffers on CPU to store intermediate results
    if store_history:
        pos_history = torch.zeros((steps + 1, N, 3), dtype=torch.float32)
        vel_history = torch.zeros((steps + 1, N, 3), dtype=torch.float32)
        # Store initial state
        pos_history[0] = pos.cpu()
        vel_history[0] = vel.cpu()
    else:
        pos_history = None
        vel_history = None

    # Pre-calculate constants
    dt_tensor = torch.tensor(dt, device=device, dtype=torch.float32)
    dt2_half = 0.5 * dt_tensor * dt_tensor
    dt_half = 0.5 * dt_tensor
    inv_m = 1.0 / mass.unsqueeze(1)

    with torch.no_grad():
        # Initial Force
        nvtx.range_push("warmup_compile")

        force_old = compute_force_func(pos, mass, G, EPSILON).clone()
        
        torch.cuda.synchronize() # Ensure compilation is done before closing range
        nvtx.range_pop()

        for step in range(steps):
            nvtx.range_push("nbody_step")

            # Update position
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)

            # Update forces
            force_new = compute_force_func(pos, mass, G, EPSILON).clone()

            # Update velocity
            vel += (force_old + force_new) * inv_m * dt_half

            # Store intermediate values
            if store_history:
                pos_history[step + 1] = pos.cpu()
                vel_history[step + 1] = vel.cpu()

            # Swap references
            force_old = force_new

            nvtx.range_pop()

    if store_history:
        return pos_history.numpy(), vel_history.numpy()
    else:
        return pos.cpu().numpy(), vel.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch N-Body Simulation")
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of particles")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-t", "--triton-block_size", type=int, default=1024, help="Block size for Triton")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    args = parser.parse_args()

    set_triton_config(args.triton_block_size)

    pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
    vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
    mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4
    
    print(f"Simulation with Pytorch. Initializing {args.num_bodies} bodies...")

    run_simulation_torch(pos, vel, mass, args.dt, args.steps, store_history=False)
    
    print("Simulation step complete.")