import torch
import numpy as np

# Constants
G = 6.67430e-11
EPSILON = 1e-4

def compute_forces_pytorch(pos, mass, G, EPSILON):
    """
    Computes gravitational forces using vectorized PyTorch operations.
    Input shapes:
      pos:  (N, 3)
      mass: (N,)
    Output:
      force: (N, 3)
    """
    # 1. Compute displacement vectors (N, N, 3)
    # Using broadcasting: (N, 1, 3) - (1, N, 3)
    # diff[i, j] = pos[i] - pos[j]
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)

    # Compute squared distances
    dist_sq = (diff ** 2).sum(dim=-1)

    # Compute inverse cubed distance with softening
    # (r^2 + epsilon^2)^(-1.5)
    dist = (dist_sq + EPSILON**2).sqrt()
    inv_dist_cube = dist.pow(-3)

    # Handle self-interaction (diagonal) to avoid NaNs if epsilon=0
    inv_dist_cube.fill_diagonal_(0.0)

    # 5. Compute acceleration contribution from j on i
    # Formula components: (r_i - r_j) * m_j / |r|^3
    # mass shape needs to be (1, N) to broadcast across columns j
    mass_j = mass.unsqueeze(0)
    
    # We sum over j (dim 1) to get the total effect on i
    # sum( diff_ij * (m_j * inv_dist_cube_ij) )
    # We use unsqueeze on the scalar factor to make it (N, N, 1) to multiply (N, N, 3)
    scalar_factor = (mass_j * inv_dist_cube).unsqueeze(-1)
    
    # Sum over j
    ftmp = (diff * scalar_factor).sum(dim=1)

    # Force = -G * m_i * ftmp
    # mass needs to be (N, 1) for i
    force = -G * mass.unsqueeze(1) * ftmp
    
    return force

def run_simulation_torch(pos_host, vel_host, mass_host, dt, steps, store_history=True):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} (PyTorch). N={pos_host.shape[0]}, Steps={steps}")

    # Move data to GPU
    pos = torch.tensor(pos_host, device=device, dtype=torch.float64)
    vel = torch.tensor(vel_host, device=device, dtype=torch.float64)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float64)

    N = pos.shape[0]

    # Allocate history buffers on CPU to store intermediate results
    if store_history:
        pos_history = torch.zeros((steps + 1, N, 3), dtype=torch.float64)
        vel_history = torch.zeros((steps + 1, N, 3), dtype=torch.float64)
        # Store initial state
        pos_history[0] = pos.cpu()
        vel_history[0] = vel.cpu()
    else:
        pos_history = None
        vel_history = None

    # Pre-calculate constants
    dt_tensor = torch.tensor(dt, device=device, dtype=torch.float64)
    dt2_half = 0.5 * dt_tensor * dt_tensor
    dt_half = 0.5 * dt_tensor
    inv_m = 1.0 / mass.unsqueeze(1)

    with torch.no_grad():
        # Initial Force
        force_old = compute_forces_pytorch(pos, mass, G, EPSILON)

        for step in range(steps):
            # Update position
            # r(t+dt) = r(t) + v(t)dt + 0.5 * F(t)/m * dt^2
            pos += (vel * dt_tensor) + (force_old * inv_m * dt2_half)

            # Upate forces
            force_new = compute_forces_pytorch(pos, mass, G, EPSILON)

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
    num_bodies = 2000
    pos = np.random.rand(num_bodies, 3).astype(np.float64) * 100.0
    vel = np.random.rand(num_bodies, 3).astype(np.float64) - 0.5
    mass = np.random.rand(num_bodies).astype(np.float64) * 1e4
    
    dt = 0.01
    steps = 100

    hist_pos, hist_vel = run_simulation_torch(pos, vel, mass, dt, steps)
    
    print("Simulation step complete.")