import torch
import triton
import triton.language as tl
import numpy as np
import argparse
import time

# Constants
G_CONST = 6.67430e-11
EPSILON_CONST = 1e-4

@triton.jit
def nbody_kernel_fast(
    pos_ptr, mass_ptr, out_ptr,
    G, EPS, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < N

    # 1. Load targets
    px = tl.load(pos_ptr + row_offsets * 3 + 0, mask=mask)
    py = tl.load(pos_ptr + row_offsets * 3 + 1, mask=mask)
    pz = tl.load(pos_ptr + row_offsets * 3 + 2, mask=mask)

    acc = tl.zeros([BLOCK_SIZE, 16], dtype=tl.float32)
    cols = tl.arange(0, 16)

    for j_start in range(0, N, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < N

        # Load sources
        qx = tl.load(pos_ptr + j_offsets * 3 + 0, mask=j_mask)
        qy = tl.load(pos_ptr + j_offsets * 3 + 1, mask=j_mask)
        qz = tl.load(pos_ptr + j_offsets * 3 + 2, mask=j_mask)
        mj = tl.load(mass_ptr + j_offsets, mask=j_mask)

        # Pad Source Matrix
        q_mat = tl.zeros([BLOCK_SIZE, 16], dtype=tl.float32)
        q_mat = tl.where(cols[None, :] == 0, qx[:, None], q_mat)
        q_mat = tl.where(cols[None, :] == 1, qy[:, None], q_mat)
        q_mat = tl.where(cols[None, :] == 2, qz[:, None], q_mat)

        # Distance calculation
        dx = qx[None, :] - px[:, None]
        dy = qy[None, :] - py[:, None]
        dz = qz[None, :] - pz[:, None]
        
        # Softened distance squared
        dist_sq = dx*dx + dy*dy + dz*dz + (EPS * EPS)
        
        # ESSENTIAL: Mask self-interaction (where i == j)
        # We check if the row index matches the column index
        is_self = row_offsets[:, None] == j_offsets[None, :]
        
        # If it's self, make distance huge so force is 0
        dist_sq = tl.where(is_self, 1e18, dist_sq)
        
        inv_dist_cube = tl.extra.cuda.libdevice.pow(dist_sq, -1.5)
        weights = (mj[None, :] * inv_dist_cube) * G
        
        # Force accumulation using Tensor Cores
        acc += tl.dot(weights, q_mat)
        
        # Second term of the expansion
        w_sum = tl.sum(weights, axis=1)[:, None]
        acc = tl.where(cols[None, :] == 0, acc - px[:, None] * w_sum, acc)
        acc = tl.where(cols[None, :] == 1, acc - py[:, None] * w_sum, acc)
        acc = tl.where(cols[None, :] == 2, acc - pz[:, None] * w_sum, acc)

    # Collapse back to 1D
    final_ax = tl.sum(tl.where(cols[None, :] == 0, acc, 0.0), axis=1)
    final_ay = tl.sum(tl.where(cols[None, :] == 1, acc, 0.0), axis=1)
    final_az = tl.sum(tl.where(cols[None, :] == 2, acc, 0.0), axis=1)

    tl.store(out_ptr + row_offsets * 3 + 0, final_ax, mask=mask)
    tl.store(out_ptr + row_offsets * 3 + 1, final_ay, mask=mask)
    tl.store(out_ptr + row_offsets * 3 + 2, final_az, mask=mask)

def compute_forces_triton(pos, mass, G, EPSILON):
    N = pos.shape[0]
    accel_out = torch.empty_like(pos)
    BLOCK_SIZE = 32 
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    nbody_kernel_fast[grid](
        pos, mass, accel_out,
        G, EPSILON, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8 # High occupancy for N=100k
    )
    return accel_out

def run_simulation_triton(pos_host, vel_host, mass_host, dt, steps, store_history=False):
    device = torch.device("cuda")
    N = pos_host.shape[0]
    
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        pos_history[0], vel_history[0] = pos_host, vel_host
    else:
        pos_history, vel_history = None, None

    # Move to GPU
    pos = torch.from_numpy(pos_host).to(device)
    vel = torch.from_numpy(vel_host).to(device)
    mass = torch.from_numpy(mass_host).to(device)

    dt2_half = 0.5 * dt * dt
    dt_half = 0.5 * dt

    # Initial force
    acc_old = compute_forces_triton(pos, mass, G_CONST, EPSILON_CONST)
    
    for step in range(steps):
        # Velocity Verlet: pos(t+dt) = pos(t) + v(t)dt + 0.5*a(t)dt^2
        pos += (vel * dt) + (acc_old * dt2_half)

        # Update Acceleration
        acc_new = compute_forces_triton(pos, mass, G_CONST, EPSILON_CONST)

        # Velocity Verlet: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))dt
        vel += (acc_old + acc_new) * dt_half

        if store_history:
            pos_history[step + 1] = pos.cpu().numpy()
            vel_history[step + 1] = vel.cpu().numpy()

        acc_old = acc_new

    if store_history:
        return pos_history, vel_history
    else:
        return pos.cpu().numpy(), vel.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton N-Body Simulation")
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of particles")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step")
    args = parser.parse_args()

    pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
    vel = (np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5) * 10.0
    mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e10 # Larger mass for visible gravity
    
    print(f"Running on GPU (Triton). N={args.num_bodies}, Steps={args.steps}")
    
    run_simulation_triton(pos, vel, mass, args.dt, args.steps, store_history=False)
    
    print("Simulation complete.")