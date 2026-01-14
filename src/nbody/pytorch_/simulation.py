import torch
import numpy as np
import argparse

# Constants
G = 6.67430e-11
EPSILON = 1e-4

# Separate the calculation into smaller chunks to fit into RAM
def compute_forces_pytorch_chunked_(pos, mass, G, EPSILON, chunk_size=128):
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

# IMPORTANT: nvcc must be available on the system before running this
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
def compute_forces_pytorch_(pos, mass, G, EPSILON):
    """
    Computes gravitational forces using vectorized PyTorch operations.
    Input shapes:
      pos:  (N, 3)
      mass: (N,)
    Output:
      force: (N, 3)
    """
    # Compute displacement vectors (N, N, 3)
    # Using broadcasting: (N, 1, 3) - (1, N, 3)
    # diff[i, j] = pos[i] - pos[j]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)

    # Compute squared distances
    dist_sq = (diff ** 2).sum(dim=-1)

    # Compute inverse cubed distance with softening
    # (r^2 + epsilon^2)^(-1.5)
    dist = (dist_sq + EPSILON**2).sqrt()
    inv_dist_cube = dist.pow(-3)

    # Compute acceleration contribution from j on i
    # Formula components: (r_i - r_j) * m_j / |r|^3
    # mass shape needs to be (1, N) to broadcast across columns j
    mass_j = mass.unsqueeze(0)
    
    # Sum over j (dim 1) to get the total effect on i
    # sum( diff_ij * (m_j * inv_dist_cube_ij) )
    # We use unsqueeze on the scalar factor to make it (N, N, 1) to multiply (N, N, 3)
    scalar_factor = (mass_j * inv_dist_cube).unsqueeze(-1)
    
    # Sum over j
    ftmp = (diff * scalar_factor).sum(dim=1)

    # Force = -G * m_i * ftmp
    # mass needs to be (N, 1) for i
    force = -G * mass.unsqueeze(1) * ftmp
    
    return force

compute_forces_pytorch = torch.compile(compute_forces_pytorch_) # JIT compile the Pytorch code
compute_forces_pytorch_chunked = torch.compile(compute_forces_pytorch_chunked_) # JIT compile the Pytorch code

def run_simulation_torch(pos_host, vel_host, mass_host, dt, steps, compute_force_func=compute_forces_pytorch, store_history=True):
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
        force_old = compute_force_func(pos, mass, G, EPSILON)

        for step in range(steps):
            # Update position
            # r(t+dt) = r(t) + v(t)dt + 0.5 * F(t)/m * dt^2
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)

            # Upate forces
            force_new = compute_force_func(pos, mass, G, EPSILON)

            # Update velocity
            # v(t+dt) = v(t) + 0.5 * (F(t) + F(t+dt))/m * dt
            vel += (force_old + force_new) * inv_m * dt_half

            # Store history
            if store_history:
                pos_history[step + 1] = pos.cpu()
                vel_history[step + 1] = vel.cpu()

            # Swap references
            force_old = force_new

    if store_history:
        return pos_history.numpy(), vel_history.numpy()
    else:
        return pos.cpu().numpy(), vel.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch N-Body Simulation")
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of particles")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    args = parser.parse_args()

    pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
    vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
    mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4
    
    print(f"Simulation with Pytorch. Initializing {args.num_bodies} bodies...")

    run_simulation_torch(pos, vel, mass, args.dt, args.steps, store_history=False)
    
    print("Simulation step complete.")