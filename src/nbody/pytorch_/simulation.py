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

def run_simulation_torch(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_pytorch_naive, store_history=False):
    # --- SETUP & TRANSFER ---
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")   
    print(f"Running on GPU (PyTorch). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")

    # torch.cuda.empty_cache()

    pos  = torch.tensor(pos_host,  device=device, dtype=torch.float32)
    vel  = torch.tensor(vel_host,  device=device, dtype=torch.float32)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float32)
    N    = pos.shape[0]

    # --- CONSTANTS PREP ---
    dt_tensor = torch.tensor(dt, device=device, dtype=torch.float32)
    dt2_half  = 0.5 * dt_tensor * dt_tensor
    dt_half   = 0.5 * dt_tensor
    inv_m     = 1.0 / mass.unsqueeze(1)
    
    # History buffer
    if store_history:
        pos_history = torch.zeros((steps + 1, N, 3), dtype=torch.float32)
        vel_history = torch.zeros((steps + 1, N, 3), dtype=torch.float32)
        pos_history[0], vel_history[0] = pos.cpu(), vel.cpu()
    else:
        pos_history, vel_history = None, None

    with torch.no_grad():
        nvtx.range_push("warmup_compile")
        
        # Initial force calculation
        force_old = compute_forces_func(pos, mass, G, EPSILON).clone()
        
        torch.cuda.synchronize()
        nvtx.range_pop()

    # --- START SIMULATION ---
    with torch.no_grad():     
        for step in range(steps):
            nvtx.range_push("nbody_step")

            # [Step A] Update Position
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)

            # [Step B] Compute Forces
            force_new = compute_forces_func(pos, mass, G, EPSILON).clone()

            # [Step C] Update Velocity
            vel += (force_old + force_new) * inv_m * dt_half
            
            if store_history:
                pos_history[step + 1] = pos.cpu()
                vel_history[step + 1] = vel.cpu()
            
            # [Step D] Swap References
            force_old = force_new

            nvtx.range_pop()
        # ===================================================

    # --- FINALIZE ---
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