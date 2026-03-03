import torch
import triton
import triton.language as tl
import numpy as np
import argparse

# Constants
G = 6.67430e-11
EPSILON = 1e-4


@triton.heuristics({
    'BLOCK_SIZE': lambda args: 128 if args['N'] < 10000 else (256 if args['N'] < 100000 else 512),
    'num_warps':  lambda args: 4   if args['N'] < 10000 else (8   if args['N'] < 100000 else 16),
    'num_stages': lambda args: 4   if args['N'] < 10000 else (3   if args['N'] < 100000 else 2)
})
@triton.jit
def compute_accel_triton_optimized(
    x_ptr, y_ptr, z_ptr, mass_ptr,            
    out_x_ptr, out_y_ptr, out_z_ptr,          
    G, EPS_SQ, N,                             
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = i_offsets < N

    # Coalesced Loads
    px_i = tl.load(x_ptr + i_offsets, mask=mask)
    py_i = tl.load(y_ptr + i_offsets, mask=mask)
    pz_i = tl.load(z_ptr + i_offsets, mask=mask)

    ax_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    ay_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    az_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for j in range(0, N):
        px_j = tl.load(x_ptr + j)
        py_j = tl.load(y_ptr + j)
        pz_j = tl.load(z_ptr + j)
        m_j  = tl.load(mass_ptr + j)

        dx = px_j - px_i
        dy = py_j - py_i
        dz = pz_j - pz_i
        
        # EPS_SQ pre-calculated
        d2 = dx*dx + dy*dy + dz*dz + EPS_SQ
        inv_dist = tl.extra.cuda.libdevice.rsqrt(d2)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        
        mass_inv_dist3 = m_j * inv_dist3
        ax_sum += dx * mass_inv_dist3
        ay_sum += dy * mass_inv_dist3
        az_sum += dz * mass_inv_dist3

    # Coalesced Stores
    tl.store(out_x_ptr + i_offsets, G * ax_sum, mask=mask)
    tl.store(out_y_ptr + i_offsets, G * ay_sum, mask=mask)
    tl.store(out_z_ptr + i_offsets, G * az_sum, mask=mask)


@triton.heuristics({
    'BLOCK_SIZE': lambda args: 128 if args['N'] < 10000 else (256 if args['N'] < 100000 else 512),
    'num_warps':  lambda args: 4   if args['N'] < 10000 else (8   if args['N'] < 100000 else 16)
})
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
    

def compute_forces_optim(pos, mass, force, G, EPSILON, N, grid_fn):
    pos_x = pos[:, 0].contiguous()
    pos_y = pos[:, 1].contiguous()
    pos_z = pos[:, 2].contiguous()

    out_x = torch.empty_like(pos_x)
    out_y = torch.empty_like(pos_y)
    out_z = torch.empty_like(pos_z)

    EPS_SQ = EPSILON * EPSILON

    compute_accel_triton_optimized[grid_fn](
        pos_x, pos_y, pos_z, mass,
        out_x, out_y, out_z,
        G, EPS_SQ, N
    )

    force[:] = torch.stack([out_x, out_y, out_z], dim=1)


def run_simulation_triton(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_accel_triton_naive, block_size=32, store_history=False):
    # --- SETUP & TRANSFER ---
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")   
    N = pos_host.shape[0]

    if hasattr(compute_forces_func, 'fn'):
        func_name = compute_forces_func.fn.__name__
    else:
        func_name = compute_forces_func.__name__

    print(f"Running on GPU (Triton). N={N}, Steps={steps}")
    print(f"Using Force Function: {func_name}")

    pos  = torch.tensor(pos_host,  device=device, dtype=torch.float32)
    vel  = torch.tensor(vel_host,  device=device, dtype=torch.float32)
    mass = torch.tensor(mass_host, device=device, dtype=torch.float32)
    force_old = torch.empty_like(pos)
    force_new = torch.empty_like(pos)

    use_optim = "optim" in func_name

    # --- CONSTANTS PREP ---
    dt_vec   = torch.tensor(dt, device=device, dtype=torch.float32)
    dt2_half = 0.5 * dt_vec * dt_vec
    dt_half  = 0.5 * dt_vec

    # Define a dynamic grid function for the autotuner
    grid_fn = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        
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

    if use_optim:
        compute_forces_optim(pos, mass, force_old, G, EPSILON, N, grid_fn)
    else:
        compute_forces_func[grid_fn](pos, mass, force_old, G, EPSILON, N)

    # --- START SIMULATION ---
    for step in range(steps):
        
        # [Step A] Update Position: r(t+dt) = r(t) + v(t)dt + 0.5*a(t)dt^2
        pos += (vel * dt_vec) + (force_old * dt2_half)

        # [Step B] Compute Acceleration: a(t+dt)
        if use_optim:
            compute_forces_optim(pos, mass, force_new, G, EPSILON, N, grid_fn)
        else:
            compute_forces_func[grid_fn](pos, mass, force_new, G, EPSILON, N)

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
    
    run_simulation_triton(pos, vel, mass, args.dt, args.steps, compute_forces_func=compute_accel_triton_naive)
    
    print("Simulation complete.")