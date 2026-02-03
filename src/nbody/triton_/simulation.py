import torch
import triton
import triton.language as tl
import numpy as np
import argparse

# Constants
G = 6.67430e-11
EPSILON = 1e-4

@triton.jit
def compute_accel_triton_naive(
    pos_ptr, mass_ptr, out_ptr,
    G, EPS, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i_offsets < N

    # Load particle i positions
    px_i = tl.load(pos_ptr + i_offsets * 3 + 0, mask=mask)
    py_i = tl.load(pos_ptr + i_offsets * 3 + 1, mask=mask)
    pz_i = tl.load(pos_ptr + i_offsets * 3 + 2, mask=mask)

    # Accumulators for partial acceleration sums (before multiplying by G)
    ax_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    ay_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    az_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for j in range(0, N):
        # Load particle j data
        px_j = tl.load(pos_ptr + j * 3 + 0)
        py_j = tl.load(pos_ptr + j * 3 + 1)
        pz_j = tl.load(pos_ptr + j * 3 + 2)
        m_j = tl.load(mass_ptr + j)

        # Vector from i to j
        dx = px_j - px_i
        dy = py_j - py_i
        dz = pz_j - pz_i
        
        # Physics: a = sum( m_j * r_vec / |r|^3 )
        d2 = dx*dx + dy*dy + dz*dz + EPS * EPS
        inv_dist = tl.extra.cuda.libdevice.rsqrt(d2)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        
        # Accumulate mass-weighted direction
        mass_inv_dist3 = m_j * inv_dist3
        ax_sum += dx * mass_inv_dist3
        ay_sum += dy * mass_inv_dist3
        az_sum += dz * mass_inv_dist3

    # Apply G once at the end
    tl.store(out_ptr + i_offsets * 3 + 0, G * ax_sum, mask=mask)
    tl.store(out_ptr + i_offsets * 3 + 1, G * ay_sum, mask=mask)
    tl.store(out_ptr + i_offsets * 3 + 2, G * az_sum, mask=mask)
    
@triton.jit
def compute_accel_triton_tensor(
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
        
        # Mask self-interaction (where i == j)
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

@triton.jit
def compute_accel_triton_tiled(
    pos_ptr, mass_ptr, out_ptr,
    G, EPS, N,
    BLOCK_SIZE: tl.constexpr
):
    # 1. Identify the "Target" particles (Rows) this block will handle
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < N

    # 2. Load Target Positions into Registers/SRAM
    # We keep these resident for the entire duration of the kernel
    x_i = tl.load(pos_ptr + rows * 3 + 0, mask=row_mask, other=0.0)
    y_i = tl.load(pos_ptr + rows * 3 + 1, mask=row_mask, other=0.0)
    z_i = tl.load(pos_ptr + rows * 3 + 2, mask=row_mask, other=0.0)

    # 3. Initialize Force Accumulators
    acc_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_y = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_z = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 4. Iterate over "Source" particles (Columns) in tiled chunks
    # This loop allows us to reuse the 'I' data against many chunks of 'J'
    for j_start in range(0, N, BLOCK_SIZE):
        cols = j_start + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < N

        # Load Source Block
        x_j = tl.load(pos_ptr + cols * 3 + 0, mask=col_mask, other=0.0)
        y_j = tl.load(pos_ptr + cols * 3 + 1, mask=col_mask, other=0.0)
        z_j = tl.load(pos_ptr + cols * 3 + 2, mask=col_mask, other=0.0)
        m_j = tl.load(mass_ptr + cols, mask=col_mask, other=0.0)

        # 5. Compute Interactions using Broadcasting
        # x_j is shape [1, BLOCK], x_i is shape [BLOCK, 1]
        # This creates a dense [BLOCK, BLOCK] matrix of interactions
        dx = x_j[None, :] - x_i[:, None]
        dy = y_j[None, :] - y_i[:, None]
        dz = z_j[None, :] - z_i[:, None]

        # Physics: r^2 + eps^2
        d2 = dx*dx + dy*dy + dz*dz + EPS*EPS
        
        # r^-1
        inv_dist = tl.extra.cuda.libdevice.rsqrt(d2)
        # r^-3
        inv_dist3 = inv_dist * inv_dist * inv_dist
        
        # Gravity Term: (m_j / r^3)
        w = m_j[None, :] * inv_dist3

        # 6. Accumulate Forces
        # We sum across the columns (axis=1) to collapse the J-block into a single force vector for I
        acc_x += tl.sum(dx * w, axis=1)
        acc_y += tl.sum(dy * w, axis=1)
        acc_z += tl.sum(dz * w, axis=1)

    # 7. Apply G and Store
    tl.store(out_ptr + rows * 3 + 0, acc_x * G, mask=row_mask)
    tl.store(out_ptr + rows * 3 + 1, acc_y * G, mask=row_mask)
    tl.store(out_ptr + rows * 3 + 2, acc_z * G, mask=row_mask)

import torch
import triton
import triton.language as tl

@triton.jit
def compute_accel_triton_mixed(
    pos_ptr, mass_ptr, out_ptr,
    G, EPS, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < N

    # LOAD IN FP16 (Half Precision) to save bandwidth/registers
    # We cast pointers to ensure Triton loads 16-bit values
    # Note: The input tensors must be cast to torch.float16 in Python before calling!
    
    # Load i-particles (Targets)
    # We keep these in registers as FP32 for better precision during math
    x_i = tl.load(pos_ptr + rows * 3 + 0, mask=row_mask, other=0.0).to(tl.float32)
    y_i = tl.load(pos_ptr + rows * 3 + 1, mask=row_mask, other=0.0).to(tl.float32)
    z_i = tl.load(pos_ptr + rows * 3 + 2, mask=row_mask, other=0.0).to(tl.float32)

    # Accumulators in FP32 (Crucial for accuracy)
    acc_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_y = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_z = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for j_start in range(0, N, BLOCK_SIZE):
        cols = j_start + tl.arange(0, BLOCK_SIZE)
        col_mask = cols < N

        # Load j-particles (Sources)
        # Load as FP16, immediately promote to FP32 for the calculation
        x_j = tl.load(pos_ptr + cols * 3 + 0, mask=col_mask, other=0.0).to(tl.float32)
        y_j = tl.load(pos_ptr + cols * 3 + 1, mask=col_mask, other=0.0).to(tl.float32)
        z_j = tl.load(pos_ptr + cols * 3 + 2, mask=col_mask, other=0.0).to(tl.float32)
        m_j = tl.load(mass_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        # Computation (Dense Broadcast)
        dx = x_j[None, :] - x_i[:, None]
        dy = y_j[None, :] - y_i[:, None]
        dz = z_j[None, :] - z_i[:, None]

        d2 = dx*dx + dy*dy + dz*dz + EPS*EPS
        inv_dist = tl.extra.cuda.libdevice.rsqrt(d2)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        w = m_j[None, :] * inv_dist3

        acc_x += tl.sum(dx * w, axis=1)
        acc_y += tl.sum(dy * w, axis=1)
        acc_z += tl.sum(dz * w, axis=1)

    # Store result in FP32
    tl.store(out_ptr + rows * 3 + 0, acc_x * G, mask=row_mask)
    tl.store(out_ptr + rows * 3 + 1, acc_y * G, mask=row_mask)
    tl.store(out_ptr + rows * 3 + 2, acc_z * G, mask=row_mask)  

def run_simulation_triton(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_accel_triton_mixed, block_size=32, store_history=False):
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
        
    # Optional history buffer
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        pos_history[0] = pos_host.copy()
        vel_history[0] = vel_host.copy()
    else:
        pos_history, vel_history = None, None

    # WARNING: Although named 'force_old/new' and 'compute_forces_func' to match other backends,
    # the Triton kernel calculates ACCELERATION directly (Force / Mass).
    # Therefore, we DO NOT multiply by inv_m in the update steps below.

    if use_mixed:
        compute_forces_func[grid](pos.to(torch.float16), mass_16, force_old, G, EPSILON, N, BLOCK_SIZE=block_size)
    else:
        compute_forces_func[grid](pos, mass, force_old, G, EPSILON, N, BLOCK_SIZE=block_size)

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

        if store_history:
            pos_history[step + 1] = pos.cpu().numpy()
            vel_history[step + 1] = vel.cpu().numpy()

        # [Step D] Swap References
        force_old, force_new = force_new, force_old
    # ===================================================

    # --- FINALIZE ---
    if store_history:
        return pos_history, vel_history
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

    print(f"Simulation with Triton. Initializing {args.num_bodies} bodies...")
    
    run_simulation_triton(pos, vel, mass, args.dt, args.steps)
    
    print("Simulation complete.")