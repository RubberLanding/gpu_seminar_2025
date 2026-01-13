import numpy as np
from numba import njit, prange, float64, cuda
import math

# --- Constants ---
G = 6.6743e-11   # Gravitational Constant (m^3 kg^-1 s^-2)
EPSILON = 1e-5

def is_gpu_available():
    return cuda.is_available()

# --- CPU KERNELS ---
@njit(parallel=True, fastmath=True)
def cpu_force_kernel_numba(r_pos, masses, r_force):
    """
    Computes gravitational forces on CPU using O(N^2) all-to-all algorithm.
    Parallelized via OpenMP (prange).
    """
    N = r_pos.shape[0]
    r_force[:] = 0.0
    
    for i in prange(N):
        ftmp_x = ftmp_y = ftmp_z = 0.0
        r_i_x, r_i_y, r_i_z = r_pos[i, 0], r_pos[i, 1], r_pos[i, 2]
        m_i = masses[i]

        for j in range(N):
            dx = r_i_x - r_pos[j, 0]
            dy = r_i_y - r_pos[j, 1]
            dz = r_i_z - r_pos[j, 2]
            
            d2 = dx*dx + dy*dy + dz*dz
            
            # Plummer Softening: prevents division by zero if particles overlap
            dist = math.sqrt(d2 + EPSILON**2)
            
            # Force Magnitude: F = (G * m_i * m_j) / dist^3 * vector_r
            F_ij_factor = masses[j] / (dist * dist * dist)
            
            ftmp_x += dx * F_ij_factor
            ftmp_y += dy * F_ij_factor
            ftmp_z += dz * F_ij_factor

        r_force[i, 0] = -G * m_i * ftmp_x
        r_force[i, 1] = -G * m_i * ftmp_y
        r_force[i, 2] = -G * m_i * ftmp_z

@njit(parallel=True, fastmath=True)
def cpu_step_pos(r_pos, v_vel, masses, F_old, dt):
    """
    Velocity Verlet Step 1: Update Position.
    r(t+dt) = r(t) + v(t)dt + 0.5 * a(t)dt^2
    """
    N = r_pos.shape[0]
    dt2_half = 0.5 * dt * dt
    for i in prange(N):
        inv_m = 1.0 / masses[i]
        r_pos[i, 0] += v_vel[i, 0] * dt + (F_old[i, 0] * inv_m) * dt2_half
        r_pos[i, 1] += v_vel[i, 1] * dt + (F_old[i, 1] * inv_m) * dt2_half
        r_pos[i, 2] += v_vel[i, 2] * dt + (F_old[i, 2] * inv_m) * dt2_half

@njit(parallel=True, fastmath=True)
def cpu_step_vel(v_vel, masses, F_old, F_new, dt):
    """
    Velocity Verlet Step 2: Update Velocity.
    v(t+dt) = v(t) + 0.5 * (a(t) + a(t+dt)) * dt
    """
    N = v_vel.shape[0]
    dt_half = 0.5 * dt
    for i in prange(N):
        inv_m = 1.0 / masses[i]
        v_vel[i, 0] += (F_old[i, 0] + F_new[i, 0]) * inv_m * dt_half
        v_vel[i, 1] += (F_old[i, 1] + F_new[i, 1]) * inv_m * dt_half
        v_vel[i, 2] += (F_old[i, 2] + F_new[i, 2]) * inv_m * dt_half

# --- GPU KERNELS ---
if is_gpu_available():
    @cuda.jit
    def gpu_force_kernel_numba(r_pos, masses, r_force):
        """CUDA Kernel for O(N^2) force calculation. One thread per particle."""
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

    @cuda.jit
    def gpu_step_pos(r_pos, v_vel, masses, F_old, dt):
        """CUDA Kernel for Position Update (Verlet Step 1)."""
        i = cuda.grid(1)
        if i < r_pos.shape[0]:
            inv_m = 1.0 / masses[i]
            dt2_half = 0.5 * dt * dt
            
            r_pos[i, 0] += v_vel[i, 0] * dt + (F_old[i, 0] * inv_m) * dt2_half
            r_pos[i, 1] += v_vel[i, 1] * dt + (F_old[i, 1] * inv_m) * dt2_half
            r_pos[i, 2] += v_vel[i, 2] * dt + (F_old[i, 2] * inv_m) * dt2_half

    @cuda.jit
    def gpu_step_vel(v_vel, masses, F_old, F_new, dt):
        """CUDA Kernel for Velocity Update (Verlet Step 2)."""
        i = cuda.grid(1)
        if i < v_vel.shape[0]:
            inv_m = 1.0 / masses[i]
            dt_half = 0.5 * dt
            
            v_vel[i, 0] += (F_old[i, 0] + F_new[i, 0]) * inv_m * dt_half
            v_vel[i, 1] += (F_old[i, 1] + F_new[i, 1]) * inv_m * dt_half
            v_vel[i, 2] += (F_old[i, 2] + F_new[i, 2]) * inv_m * dt_half

def run_simulation_numba(r_pos_host, v_vel_host, masses_host, dt, steps, device="auto", store_history=True):
    """
    Run the N-body simulation using Numba.
    
    Args:
        device (str): "cpu", "gpu", or "auto".
        store_history (bool): If True, returns (steps, N, 3) array. 
                              If False, returns final state (pos, vel).
    """
    N = r_pos_host.shape[0]
    use_gpu = (device == "gpu" or device == "auto") and is_gpu_available()
    
    # Allocate history buffers on CPU to store intermediate results
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float64)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float64)
        # Store initial state
        pos_history[0] = r_pos_host.copy()
        vel_history[0] = v_vel_host.copy()
    else:
        pos_history = None
        vel_history = None

    # Run on GPU
    if use_gpu:
        print(f"Running on GPU (Numba). N={N}, Steps={steps}")
        threads = 256
        blocks = math.ceil(N / threads)

        # Move data to GPU
        d_pos = cuda.to_device(r_pos_host)
        d_vel = cuda.to_device(v_vel_host)
        d_mass = cuda.to_device(masses_host)

        # Allocate force buffers on GPU
        d_F_old = cuda.device_array((N, 3), dtype=np.float64)
        d_F_new = cuda.device_array((N, 3), dtype=np.float64)
        
        # Initial force calculation
        gpu_force_kernel_numba[blocks, threads](d_pos, d_mass, d_F_old)
        
        for step in range(steps):
            # Update position
            gpu_step_pos[blocks, threads](d_pos, d_vel, d_mass, d_F_old, dt)
            
            # Update force
            gpu_force_kernel_numba[blocks, threads](d_pos, d_mass, d_F_new)
            
            # Update Velocity
            gpu_step_vel[blocks, threads](d_vel, d_mass, d_F_old, d_F_new, dt)

            # Store history
            if store_history:
                d_pos.copy_to_host(pos_history[step+1])
                d_vel.copy_to_host(vel_history[step+1])
            
            # Swap references
            d_F_old, d_F_new = d_F_new, d_F_old
            
        if store_history:
            return pos_history, vel_history
        else:
            return (d_pos.copy_to_host(), d_vel.copy_to_host())
        
    # Run on CPU
    else:
        print(f"Running on CPU. N={N}, Steps={steps}")
        r_pos = r_pos_host.copy()
        v_vel = v_vel_host.copy()
        masses = masses_host
        F_old = np.zeros_like(r_pos)
        F_new = np.zeros_like(r_pos)
        
        cpu_force_kernel_numba(r_pos, masses, F_old)
        
        for step in range(steps):
            cpu_step_pos(r_pos, v_vel, masses, F_old, dt)  

            cpu_force_kernel_numba(r_pos, masses, F_new)

            cpu_step_vel(v_vel, masses, F_old, F_new, dt)

            if store_history:
                pos_history[step+1] = r_pos.copy()
                vel_history[step+1] = v_vel.copy()            

            F_old[:] = F_new[:] 
            
        return (pos_history, vel_history) if store_history else (r_pos, v_vel)

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    num_bodies = 2000
    pos = np.random.rand(num_bodies, 3).astype(np.float64) * 100.0
    vel = np.random.rand(num_bodies, 3).astype(np.float64) - 0.5
    mass = np.random.rand(num_bodies).astype(np.float64) * 1e4
    
    dt = 0.01
    steps = 100
    
    res = run_simulation_numba(pos, vel, mass, dt, steps, device="gpu")
    print("Simulation step complete.")