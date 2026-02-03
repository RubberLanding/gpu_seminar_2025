import numpy as np
from numba import njit, prange, float64, float32, cuda
import math
import numba
import argparse

# --- Constants ---
G = 6.6743e-11   # Gravitational Constant (m^3 kg^-1 s^-2)
EPSILON = 1e-5

@cuda.jit(lineinfo=True, fastmath=True)
def compute_forces_numba_naive(r_pos, masses, r_force, G, EPSILON):
    N = r_pos.shape[0]
    i = cuda.grid(1)
    
    if i < N:
        ftmp_x = ftmp_y = ftmp_z = 0.0
        r_i_x, r_i_y, r_i_z = r_pos[i, 0], r_pos[i, 1], r_pos[i, 2]
        m_i = masses[i]
        
        for j in range(N):
            dx = r_i_x - r_pos[j, 0]
            dy = r_i_y - r_pos[j, 1]
            dz = r_i_z - r_pos[j, 2]
            
            d2 = dx*dx + dy*dy + dz*dz
            dist = math.sqrt(d2 + EPSILON**2) 
            
            val = masses[j] / (dist * dist * dist)
            
            ftmp_x += dx * val
            ftmp_y += dy * val
            ftmp_z += dz * val

        r_force[i, 0] = -G * m_i * ftmp_x
        r_force[i, 1] = -G * m_i * ftmp_y
        r_force[i, 2] = -G * m_i * ftmp_z

@cuda.jit(fastmath=True)
def gpu_step_pos(pos, vel, masses, F_old, dt):
    """CUDA Kernel for Position Update (Verlet Step 1)."""
    i = cuda.grid(1)
    if i < pos.shape[0]:
        inv_m = 1.0 / masses[i]
        dt2_half = 0.5 * dt * dt
        
        pos[i, 0] += vel[i, 0] * dt + (F_old[i, 0] * inv_m) * dt2_half
        pos[i, 1] += vel[i, 1] * dt + (F_old[i, 1] * inv_m) * dt2_half
        pos[i, 2] += vel[i, 2] * dt + (F_old[i, 2] * inv_m) * dt2_half

@cuda.jit(fastmath=True)
def gpu_step_vel(vel, masses, F_old, F_new, dt):
    """CUDA Kernel for Velocity Update (Verlet Step 2)."""
    i = cuda.grid(1)
    if i < vel.shape[0]:
        inv_m = 1.0 / masses[i]
        dt_half = 0.5 * dt
        
        vel[i, 0] += (F_old[i, 0] + F_new[i, 0]) * inv_m * dt_half
        vel[i, 1] += (F_old[i, 1] + F_new[i, 1]) * inv_m * dt_half
        vel[i, 2] += (F_old[i, 2] + F_new[i, 2]) * inv_m * dt_half

def compute_forces_numba_tiled(threads_per_block):
    TPB = threads_per_block

    @cuda.jit(fastmath=True)
    def compute_forces_numba_tiled_(r_pos, masses, r_force, G, EPSILON):
        # 1. Flatten Shared Memory to eliminate bank conflicts
        s_x = cuda.shared.array(shape=TPB, dtype=float32)
        s_y = cuda.shared.array(shape=TPB, dtype=float32)
        s_z = cuda.shared.array(shape=TPB, dtype=float32)
        s_m = cuda.shared.array(shape=TPB, dtype=float32)

        tx = cuda.threadIdx.x
        i = cuda.grid(1)
        N = r_pos.shape[0]

        # 2. Register Caching: Pull 'target' particle into registers once
        rx, ry, rz = 0.0, 0.0, 0.0
        acc_x, acc_y, acc_z = 0.0, 0.0, 0.0
        if i < N:
            rx, ry, rz = r_pos[i, 0], r_pos[i, 1], r_pos[i, 2]

        for tile in range((N + TPB - 1) // TPB):
            # 3. Collaborative Load
            t_idx = tile * TPB + tx
            if t_idx < N:
                s_x[tx] = r_pos[t_idx, 0]
                s_y[tx] = r_pos[t_idx, 1]
                s_z[tx] = r_pos[t_idx, 2]
                s_m[tx] = masses[t_idx]
            else:
                s_m[tx] = 0.0
                
            cuda.syncthreads() # Wait for tile load

            # Simple 1D indexing allows the compiler to use 'unroll' and pointer arithmetic
            for j in range(TPB):
                if (tile * TPB + j) < N: # Ensures we don't calculate "ghost" forces
                    dx = s_x[j] - rx
                    dy = s_y[j] - ry
                    dz = s_z[j] - rz
                    
                    dist_sq = dx*dx + dy*dy + dz*dz + EPSILON**2
                    inv_dist = 1.0 / math.sqrt(dist_sq)
                    val = s_m[j] * (inv_dist * inv_dist * inv_dist)
                    
                    acc_x += dx * val
                    acc_y += dy * val
                    acc_z += dz * val
            
            cuda.syncthreads() # Wait for all threads to finish math before next tile load

        if i < N:
            m_i = masses[i]
            r_force[i, 0] = G * m_i * acc_x
            r_force[i, 1] = G * m_i * acc_y
            r_force[i, 2] = G * m_i * acc_z
    
    compute_forces_numba_tiled_.__name__ = f"compute_forces_numba_tiled_{TPB}"
    return compute_forces_numba_tiled_

def run_simulation_numba(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_numba_naive, threads=128, store_history=False):
    # --- SETUP & TRANSFER ---
    print(f"Running on GPU (Numba). N={N}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")
    
    N = pos_host.shape[0]
    pos  = cuda.to_device(pos_host)
    vel  = cuda.to_device(vel_host)
    mass = cuda.to_device(mass_host)
    force_old = cuda.device_array((N, 3), dtype=np.float32)
    force_new = cuda.device_array((N, 3), dtype=np.float32)

    # --- CONSTANTS PREP ---
    blocks  = math.ceil(N / threads)

    if compute_forces_func == compute_forces_numba_tiled: 
        compute_forces_func = compute_forces_func(threads)

    # History buffer
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        pos_history[0] = pos_host.copy()
        vel_history[0] = vel_host.copy()
    else:
        pos_history, vel_history = None, None

    # Initial force calculation
    compute_forces_func[blocks, threads](pos, mass, force_old, G, EPSILON)
    
    # --- START SIMULATION ---
    for step in range(steps):
        
        # [Step A] Update Position: r(t+dt) = r(t) + v(t)dt + 0.5*a(t)dt^2
        gpu_step_pos[blocks, threads](pos, vel, mass, force_old, dt)
        
        # [Step B] Compute Forces: F(t+dt)
        compute_forces_func[blocks, threads](pos, mass, force_new, G, EPSILON)
        
        # [Step C] Update Velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))dt
        gpu_step_vel[blocks, threads](vel, mass, force_old, force_new, dt)
        
        if store_history:
            pos.copy_to_host(pos_history[step + 1])
            vel.copy_to_host(vel_history[step + 1])
        
        # [Step D] Swap References
        force_old, force_new = force_new, force_old
    # ===================================================

    # --- FINALIZE ---
    if store_history:
        return pos_history, vel_history
    else:
        return pos.copy_to_host(), vel.copy_to_host()
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Numba N-Body Simulation")
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of particles")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    args = parser.parse_args()

    pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
    vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
    mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4
    
    print(f"Simulation with Numba. Initializing {args.num_bodies} bodies...")

    run_simulation_numba(pos, vel, mass, args.dt, args.steps)
    
    print("Simulation step complete.")