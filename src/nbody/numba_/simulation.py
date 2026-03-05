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

@cuda.jit(lineinfo=True, fastmath=True)
def gpu_step_pos(pos, vel, masses, F_old, dt):
    """CUDA Kernel for Position Update (Verlet Step 1)."""
    i = cuda.grid(1)
    if i < pos.shape[0]:
        inv_m = 1.0 / masses[i]
        dt2_half = 0.5 * dt * dt
        
        pos[i, 0] += vel[i, 0] * dt + (F_old[i, 0] * inv_m) * dt2_half
        pos[i, 1] += vel[i, 1] * dt + (F_old[i, 1] * inv_m) * dt2_half
        pos[i, 2] += vel[i, 2] * dt + (F_old[i, 2] * inv_m) * dt2_half

@cuda.jit(lineinfo=True, fastmath=True)
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
    def compute_forces_numba_tiled_(pos_x, pos_y, pos_z, mass, 
                                force_x, force_y, force_z, 
                                N, G, EPSILON):
        
        # Allocate SoA arrays for coalesced memory access
        sx = cuda.shared.array(shape=TPB, dtype=float32)
        sy = cuda.shared.array(shape=TPB, dtype=float32)
        sz = cuda.shared.array(shape=TPB, dtype=float32)
        sm = cuda.shared.array(shape=TPB, dtype=float32)

        tx = cuda.threadIdx.x
        i = cuda.grid(1)
        rx = ry = rz = m_i = 0.0
        fx = fy = fz = 0.0
        
        if i < N:
            rx = pos_x[i]
            ry = pos_y[i]
            rz = pos_z[i]
            m_i = mass[i]

        # Loop over all tiles
        num_tiles = (N + TPB - 1) // TPB
        for t in range(num_tiles):
            
            j = t * TPB + tx
            if j < N:
                sx[tx] = pos_x[j]
                sy[tx] = pos_y[j]
                sz[tx] = pos_z[j]
                sm[tx] = mass[j]
            else:
                sx[tx] = 0.0
                sy[tx] = 0.0
                sz[tx] = 0.0
                sm[tx] = 0.0
                
            # Wait for the tile to load
            cuda.syncthreads() 

            # Compute forces
            if i < N:
                # Because TPB is constant, this loop should unroll.
                for k in range(TPB):
                    dx = sx[k] - rx
                    dy = sy[k] - ry
                    dz = sz[k] - rz

                    d2 = dx*dx + dy*dy + dz*dz + EPSILON**2
                    inv_dist = 1.0 / math.sqrt(d2) 
                    
                    inv_dist3 = inv_dist * inv_dist * inv_dist
                    s = sm[k] * inv_dist3
                    
                    fx += dx * s
                    fy += dy * s
                    fz += dz * s

            # Wait for math to finish before overwriting tile
            cuda.syncthreads() 

        # Coalesced Memory Writes
        if i < N:
            force_x[i] = fx * G * m_i
            force_y[i] = fy * G * m_i
            force_z[i] = fz * G * m_i

    compute_forces_numba_tiled_.__name__ = f"compute_forces_numba_tiled_{TPB}"
    return compute_forces_numba_tiled_

@cuda.jit
def update_position_soa(pos_x, pos_y, pos_z, 
                        vel_x, vel_y, vel_z, 
                        force_x, force_y, force_z, 
                        inv_mass, dt, N):
    i = cuda.grid(1)
    if i < N:
        inv_m = inv_mass[i]
        dt2_half = 0.5 * dt * dt
        
        pos_x[i] += (vel_x[i] * dt) + (force_x[i] * inv_m * dt2_half)
        pos_y[i] += (vel_y[i] * dt) + (force_y[i] * inv_m * dt2_half)
        pos_z[i] += (vel_z[i] * dt) + (force_z[i] * inv_m * dt2_half)

@cuda.jit
def update_velocity_soa(vel_x, vel_y, vel_z, 
                        f_old_x, f_old_y, f_old_z, 
                        f_new_x, f_new_y, f_new_z, 
                        inv_mass, dt, N):
    i = cuda.grid(1)
    if i < N:
        inv_m = inv_mass[i]
        dt_half = 0.5 * dt
        
        vel_x[i] += (f_old_x[i] + f_new_x[i]) * inv_m * dt_half
        vel_y[i] += (f_old_y[i] + f_new_y[i]) * inv_m * dt_half
        vel_z[i] += (f_old_z[i] + f_new_z[i]) * inv_m * dt_half


def run_simulation_numba(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_numba_naive, threads=128, store_history=False):
    # --- SETUP & TRANSFER ---
    N = pos_host.shape[0]
    blocks = (N + threads - 1) // threads

    # Check if the passed function is the SoA optimized version
    # getattr is used safely in case the function is wrapped/decorated
    func_name = getattr(compute_forces_func, '__name__', 'Unknown')
    is_soa = ('tiled' in func_name)

    print(f"Running on GPU (Numba). N={N}, Steps={steps}")
    print(f"Using Force Function: {func_name}")

    # History buffer
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        pos_history[0] = pos_host.copy()
        vel_history[0] = vel_host.copy()
    else:
        pos_history, vel_history = None, None

    if is_soa:
        # =========================================================
        # PATH A: STRUCTURE OF ARRAYS (SoA)
        # =========================================================
        compute_forces_func = compute_forces_func(threads)

        px = np.ascontiguousarray(pos_host[:, 0], dtype=np.float32)
        py = np.ascontiguousarray(pos_host[:, 1], dtype=np.float32)
        pz = np.ascontiguousarray(pos_host[:, 2], dtype=np.float32)
        
        vx = np.ascontiguousarray(vel_host[:, 0], dtype=np.float32)
        vy = np.ascontiguousarray(vel_host[:, 1], dtype=np.float32)
        vz = np.ascontiguousarray(vel_host[:, 2], dtype=np.float32)

        mass = np.ascontiguousarray(mass_host, dtype=np.float32)
        inv_mass = np.ascontiguousarray(1.0 / mass_host, dtype=np.float32)

        d_px, d_py, d_pz = cuda.to_device(px), cuda.to_device(py), cuda.to_device(pz)
        d_vx, d_vy, d_vz = cuda.to_device(vx), cuda.to_device(vy), cuda.to_device(vz)
        d_mass, d_inv_mass = cuda.to_device(mass), cuda.to_device(inv_mass)

        d_f_old_x = cuda.device_array(N, dtype=np.float32)
        d_f_old_y = cuda.device_array(N, dtype=np.float32)
        d_f_old_z = cuda.device_array(N, dtype=np.float32)
        
        d_f_new_x = cuda.device_array(N, dtype=np.float32)
        d_f_new_y = cuda.device_array(N, dtype=np.float32)
        d_f_new_z = cuda.device_array(N, dtype=np.float32)

        # Initial force computation
        compute_forces_func[blocks, threads](d_px, d_py, d_pz, d_mass, d_f_old_x, d_f_old_y, d_f_old_z, N, np.float32(G), np.float32(EPSILON))

        cuda.synchronize()
        cuda.profile_start()

        # --- START SIMULATION ---
        for step in range(steps):
            # [Step A] Update Position: r(t+dt) = r(t) + v(t)dt + 0.5*a(t)dt^2
            update_position_soa[blocks, threads](d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_f_old_x, d_f_old_y, d_f_old_z, d_inv_mass, np.float32(dt), N)
            
            # [Step B] Compute Forces: F(t+dt)
            compute_forces_func[blocks, threads](d_px, d_py, d_pz, d_mass, d_f_new_x, d_f_new_y, d_f_new_z, N, np.float32(G), np.float32(EPSILON))
            
            # [Step C] Update Velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))dt
            update_velocity_soa[blocks, threads](d_vx, d_vy, d_vz, d_f_old_x, d_f_old_y, d_f_old_z, d_f_new_x, d_f_new_y, d_f_new_z, d_inv_mass, np.float32(dt), N)
            
            if store_history:                
                pos_history[step + 1, :, 0] = d_px.copy_to_host()
                pos_history[step + 1, :, 1] = d_py.copy_to_host()
                pos_history[step + 1, :, 2] = d_pz.copy_to_host()

                vel_history[step + 1, :, 0] = d_vx.copy_to_host()
                vel_history[step + 1, :, 1] = d_vy.copy_to_host()
                vel_history[step + 1, :, 2] = d_vz.copy_to_host()
            
            # [Step D] Swap References
            d_f_old_x, d_f_new_x = d_f_new_x, d_f_old_x
            d_f_old_y, d_f_new_y = d_f_new_y, d_f_old_y
            d_f_old_z, d_f_new_z = d_f_new_z, d_f_old_z
        
        cuda.profile_stop()

        # --- FINALIZE ---
        if store_history:
            return pos_history, vel_history
        else:
            pos = np.zeros((N, 3), dtype=np.float32)
            vel = np.zeros((N, 3), dtype=np.float32)
            pos[:, 0] = d_px.copy_to_host()
            pos[:, 1] = d_py.copy_to_host()
            pos[:, 2] = d_pz.copy_to_host()

            vel[:, 0] = d_vx.copy_to_host()
            vel[:, 1] = d_vy.copy_to_host()
            vel[:, 2] = d_vz.copy_to_host()

            return pos, vel

    else:
        pos  = cuda.to_device(pos_host)
        vel  = cuda.to_device(vel_host)
        mass = cuda.to_device(mass_host)
        force_old = cuda.device_array((N, 3), dtype=np.float32)
        force_new = cuda.device_array((N, 3), dtype=np.float32)
        
        # Initial force calculation
        compute_forces_func[blocks, threads](pos, mass, force_old, G, EPSILON)

        cuda.synchronize()
        cuda.profile_start()
        
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

        cuda.profile_stop()

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

    run_simulation_numba(pos, vel, mass, args.dt, args.steps, compute_forces_func=compute_forces_numba_naive)
    
    print("Simulation step complete.")