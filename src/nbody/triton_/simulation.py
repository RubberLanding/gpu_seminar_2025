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
        
        # a = sum( m_j * r_vec / |r|^3 )
        d2 = dx*dx + dy*dy + dz*dz + EPS * EPS
        inv_dist = tl.extra.cuda.libdevice.rsqrt(d2)
        inv_dist3 = inv_dist * inv_dist * inv_dist
        
        mass_inv_dist3 = m_j * inv_dist3
        ax_sum += dx * mass_inv_dist3
        ay_sum += dy * mass_inv_dist3
        az_sum += dz * mass_inv_dist3

    # Apply G once at the end
    tl.store(out_ptr + i_offsets * 3 + 0, G * ax_sum, mask=mask)
    tl.store(out_ptr + i_offsets * 3 + 1, G * ay_sum, mask=mask)
    tl.store(out_ptr + i_offsets * 3 + 2, G * az_sum, mask=mask)
    


@triton.heuristics({
    'num_warps': lambda args: max(1, args['BLOCK_SIZE'] // 32)
})
@triton.jit
def compute_accel_triton_optimized(
    pos_x_ptr, pos_y_ptr, pos_z_ptr,
    mass_ptr, 
    out_x_ptr, out_y_ptr, out_z_ptr,
    G, EPS, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N

    # Coalesced SoA loads for target particles 'i'
    px_i = tl.load(pos_x_ptr + i_offsets, mask=i_mask, other=0.0)
    py_i = tl.load(pos_y_ptr + i_offsets, mask=i_mask, other=0.0)
    pz_i = tl.load(pos_z_ptr + i_offsets, mask=i_mask, other=0.0)

    ax_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    ay_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    az_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    EPS2 = EPS * EPS

    for j in range(0, N):
        px_j = tl.load(pos_x_ptr + j)
        py_j = tl.load(pos_y_ptr + j)
        pz_j = tl.load(pos_z_ptr + j)
        m_j  = tl.load(mass_ptr + j)

        dx = px_j - px_i
        dy = py_j - py_i
        dz = pz_j - pz_i
        
        # Explicitly fuse the d2 accumulation
        d2 = dx*dx + EPS2
        d2 = tl.math.fma(dy, dy, d2)
        d2 = tl.math.fma(dz, dz, d2)

        # 1-cycle hardware approximation
        inv_dist = tl.math.rsqrt(d2) 
        inv_dist3 = inv_dist * inv_dist * inv_dist
        
        mass_inv_dist3 = m_j * inv_dist3
        
        # Explicit hardware FMA
        ax_sum = tl.math.fma(dx, mass_inv_dist3, ax_sum)
        ay_sum = tl.math.fma(dy, mass_inv_dist3, ay_sum)
        az_sum = tl.math.fma(dz, mass_inv_dist3, az_sum)

    # Coalesced SoA stores
    tl.store(out_x_ptr + i_offsets, G * ax_sum, mask=i_mask)
    tl.store(out_y_ptr + i_offsets, G * ay_sum, mask=i_mask)
    tl.store(out_z_ptr + i_offsets, G * az_sum, mask=i_mask)

def run_simulation_triton(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_accel_triton_naive, block_size=32, store_history=False):
    # --- SETUP & TRANSFER ---
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.device("cuda")   
    N = pos_host.shape[0]

    if hasattr(compute_forces_func, 'fn'):
        func_name = compute_forces_func.fn.__name__
    else:
        func_name = compute_forces_func.__name__

    if 'soa' in func_name:
        is_soa = True

    print(f"Running on GPU (Triton). N={N}, Steps={steps}")
    print(f"Using Force Function: {func_name}")

    if is_soa:
        pos_x = torch.tensor(pos_host[:, 0], device=device, dtype=torch.float32).contiguous()
        pos_y = torch.tensor(pos_host[:, 1], device=device, dtype=torch.float32).contiguous()
        pos_z = torch.tensor(pos_host[:, 2], device=device, dtype=torch.float32).contiguous()

        vel_x = torch.tensor(vel_host[:, 0], device=device, dtype=torch.float32).contiguous()
        vel_y = torch.tensor(vel_host[:, 1], device=device, dtype=torch.float32).contiguous()
        vel_z = torch.tensor(vel_host[:, 2], device=device, dtype=torch.float32).contiguous()

        mass = torch.tensor(mass_host, device=device, dtype=torch.float32).contiguous()
        
        # Separate force/acceleration buffers
        force_old_x = torch.empty_like(pos_x)
        force_old_y = torch.empty_like(pos_y)
        force_old_z = torch.empty_like(pos_z)
        
        force_new_x = torch.empty_like(pos_x)
        force_new_y = torch.empty_like(pos_y)
        force_new_z = torch.empty_like(pos_z)

        # Constants prep
        dt_vec = torch.tensor(dt, device=device, dtype=torch.float32)
        dt2_half = 0.5 * dt_vec * dt_vec
        dt_half = 0.5 * dt_vec

        def grid(meta):
            return (triton.cdiv(N, meta['BLOCK_SIZE']),)
            
        if store_history:
            pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
            vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
            pos_history[0] = pos_host.copy()
            vel_history[0] = vel_host.copy()
        else:
            pos_history, vel_history = None, None

        # Initial force calculation using SoA kernel
        compute_forces_func[grid](
            pos_x, pos_y, pos_z, mass, 
            force_old_x, force_old_y, force_old_z, 
            G, EPSILON, N, BLOCK_SIZE=block_size
        )

        # --- START SIMULATION ---
        for step in range(steps):
            
            # [Step A] Update Position (SoA math)
            pos_x += (vel_x * dt_vec) + (force_old_x * dt2_half)
            pos_y += (vel_y * dt_vec) + (force_old_y * dt2_half)
            pos_z += (vel_z * dt_vec) + (force_old_z * dt2_half)

            # [Step B] Compute Acceleration
            compute_forces_func[grid](
                pos_x, pos_y, pos_z, mass, 
                force_new_x, force_new_y, force_new_z, 
                G, EPSILON, N, BLOCK_SIZE=block_size
            )

            # [Step C] Update Velocity (SoA math)
            vel_x += (force_old_x + force_new_x) * dt_half
            vel_y += (force_old_y + force_new_y) * dt_half
            vel_z += (force_old_z + force_new_z) * dt_half

            if store_history:
                # Reconstruct AoS layout for the history buffer
                pos_history[step + 1, :, 0] = pos_x.cpu().numpy()
                pos_history[step + 1, :, 1] = pos_y.cpu().numpy()
                pos_history[step + 1, :, 2] = pos_z.cpu().numpy()
                vel_history[step + 1, :, 0] = vel_x.cpu().numpy()
                vel_history[step + 1, :, 1] = vel_y.cpu().numpy()
                vel_history[step + 1, :, 2] = vel_z.cpu().numpy()

            # [Step D] Swap References
            force_old_x, force_new_x = force_new_x, force_old_x
            force_old_y, force_new_y = force_new_y, force_old_y
            force_old_z, force_new_z = force_new_z, force_old_z

        # --- FINALIZE ---
        if store_history:
            return pos_history, vel_history
            
        # Reconstruct the final (N, 3) numpy arrays to match original API
        final_pos = torch.stack([pos_x, pos_y, pos_z], dim=1).cpu().numpy()
        final_vel = torch.stack([vel_x, vel_y, vel_z], dim=1).cpu().numpy()
        return final_pos, final_vel
    
    else:
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
            compute_forces_func(pos, mass, force_old, G, EPSILON, N, grid_fn)
        else:
            compute_forces_func[grid_fn](pos, mass, force_old, G, EPSILON, N)

        # --- START SIMULATION ---
        for step in range(steps):
            
            # [Step A] Update Position: r(t+dt) = r(t) + v(t)dt + 0.5*a(t)dt^2
            pos += (vel * dt_vec) + (force_old * dt2_half)

            # [Step B] Compute Acceleration: a(t+dt)
            if use_optim:
                compute_forces_func(pos, mass, force_new, G, EPSILON, N, grid_fn)
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