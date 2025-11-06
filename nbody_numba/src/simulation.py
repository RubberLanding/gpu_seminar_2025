import numpy as np
from numba import njit, prange, float64, cuda
import math

# System Constants
G = 6.6743e-11  # Gravitational constant
EPSILON = 1e-1  # Smoothing factor (softening parameter)

def is_gpu_available():
    """Checks if Numba's CUDA target is available."""
    try:
        return cuda.is_available()
    except Exception:
        return False

# Force computation kernel for CPU
@njit(parallel=True, fastmath=True)
def nbody_cpu_kernel(r_pos, masses):
    """
    Computes N-body forces on the CPU using Numba JIT (O(N^2)).
    Parallelized with prange for multi-core performance.
    """
    N = r_pos.shape[0]
    r_force = np.zeros_like(r_pos)
    
    # prange enables parallel execution of the outer loop
    for i in prange(N):
        ftmp_x = 0.0
        ftmp_y = 0.0
        ftmp_z = 0.0
        
        r_i_x, r_i_y, r_i_z = r_pos[i, 0], r_pos[i, 1], r_pos[i, 2]
        m_i = masses[i]

        for j in range(N):
            dx = r_i_x - r_pos[j, 0]
            dy = r_i_y - r_pos[j, 1]
            dz = r_i_z - r_pos[j, 2]
            
            d2 = dx * dx + dy * dy + dz * dz
            
            # Apply smoothing term epsilon (d = sqrt(d^2) + epsilon)
            dist = math.sqrt(d2) + EPSILON
            dist3 = dist * dist * dist
            
            F_ij_factor = masses[j] / dist3
            
            ftmp_x += dx * F_ij_factor
            ftmp_y += dy * F_ij_factor
            ftmp_z += dz * F_ij_factor

        r_force[i, 0] = -G * m_i * ftmp_x
        r_force[i, 1] = -G * m_i * ftmp_y
        r_force[i, 2] = -G * m_i * ftmp_z
        
    return r_force

# Force computation kernel for GPU
if is_gpu_available():
    @cuda.jit('void(float64[:, :], float64[:], float64[:, :], float64, float64)')
    def nbody_gpu_kernel(r_pos, masses, r_force, G_const, epsilon):
        """
        CUDA kernel to compute N-body forces on the GPU (O(N^2)).
        Each thread calculates the total force on a single particle 'i'.
        """
        N = r_pos.shape[0]
        i = cuda.grid(1) 
        
        if i < N:
            ftmp_x = 0.0
            ftmp_y = 0.0
            ftmp_z = 0.0
            
            r_i_x, r_i_y, r_i_z = r_pos[i, 0], r_pos[i, 1], r_pos[i, 2]
            m_i = masses[i]
            
            for j in range(N):
                dx = r_i_x - r_pos[j, 0]
                dy = r_i_y - r_pos[j, 1]
                dz = r_i_z - r_pos[j, 2]
                
                d2 = dx * dx + dy * dy + dz * dz
                dist = math.sqrt(d2) + epsilon
                dist3 = dist * dist * dist
                F_ij_factor = masses[j] / dist3
                
                ftmp_x += dx * F_ij_factor
                ftmp_y += dy * F_ij_factor
                ftmp_z += dz * F_ij_factor

            r_force[i, 0] = -G_const * m_i * ftmp_x
            r_force[i, 1] = -G_const * m_i * ftmp_y
            r_force[i, 2] = -G_const * m_i * ftmp_z

# Wrapper to run the force computation either on CPU or GPU
def compute_nbody_force(r_pos_host, masses_host, device="auto"):
    """
    High-level function to select and run the appropriate N-body kernel.
    """
    if (device == "gpu" or device == "auto") and is_gpu_available():
        N = r_pos_host.shape[0]
        
        r_pos_device = cuda.to_device(r_pos_host)
        masses_device = cuda.to_device(masses_host)
        r_force_device = cuda.device_array_like(r_pos_host)
        
        threadsperblock = 256
        blockspergrid = math.ceil(N / threadsperblock)
        
        nbody_gpu_kernel[blockspergrid, threadsperblock](
            r_pos_device, 
            masses_device, 
            r_force_device, 
            G, 
            EPSILON
        )
        
        return r_force_device.copy_to_host()

    else:
        # Fallback or explicit CPU
        return nbody_cpu_kernel(r_pos_host, masses_host)

# Time integration kernels for updating the positions and velocities on the CPU
@njit(parallel=True, fastmath=True, cache=True)
def position_integration_kernel(r_pos, v_vel, masses, F_old, dt):
    """Updates position based on current velocity and OLD force F^t.
    x^(t+1) = x^t + v^t * dt + 0.5 * a^t * dt^2
    """
    N = r_pos.shape[0]
    dt2_half = 0.5 * dt * dt
    
    for i in prange(N):
        inv_m = 1.0 / masses[i]
        
        # Acceleration: a^t = F^t / m_i
        ax = F_old[i, 0] * inv_m
        ay = F_old[i, 1] * inv_m
        az = F_old[i, 2] * inv_m
        
        # Modified Euler 
        r_pos[i, 0] += v_vel[i, 0] * dt + ax * dt2_half
        r_pos[i, 1] += v_vel[i, 1] * dt + ay * dt2_half
        r_pos[i, 2] += v_vel[i, 2] * dt + az * dt2_half

@njit(parallel=True, fastmath=True, cache=True)
def velocity_integration_kernel(v_vel, masses, F_old, F_new, dt):
    """Updates velocity using the average of OLD and NEW forces.
    v^(t+1) = v^t + 0.5 * (a^t + a^(t+1)) * dt
    """
    N = v_vel.shape[0]
    
    for i in prange(N):
        inv_m = 1.0 / masses[i]
        
        # Modified Euler: 0.5 * (F_old + F_new) / m_i
        avg_acc_x = 0.5 * (F_old[i, 0] + F_new[i, 0]) * inv_m
        avg_acc_y = 0.5 * (F_old[i, 1] + F_new[i, 1]) * inv_m
        avg_acc_z = 0.5 * (F_old[i, 2] + F_new[i, 2]) * inv_m
        
        v_vel[i, 0] += avg_acc_x * dt
        v_vel[i, 1] += avg_acc_y * dt
        v_vel[i, 2] += avg_acc_z * dt

# Use Modified Euler for time integration 
def run_rk2_simulation(initial_positions, initial_velocities, masses, 
                         dt, n_loops, device="auto", store_history=True):
    """
    Runs the N-body simulation using the RK2 / Modified Euler scheme.
    """
    r_pos = initial_positions.copy()
    v_vel = initial_velocities.copy()

    if store_history:
        # Create an array to store positions at each time step
        # Shape: (num_steps, num_particles, 3_dimensions)
        positions_history = np.zeros((n_loops, len(masses), 3), dtype=np.float64)
    
    # Calculate initial forces (F^t)
    F_old = compute_nbody_force(r_pos, masses, device=device)
    F_new = np.zeros_like(r_pos) # Buffer for F^(t+1)
    
    print(f"Starting RK2 simulation for {len(masses)} particles. Steps: {n_loops}. Device: {device.upper()}")

    # Integration loop
    for step in range(n_loops):
        
        # Position integration:
        # x^(t+1) = x^t + v^t * dt + 0.5 * a^t * dt^2
        position_integration_kernel(r_pos, v_vel, masses, F_old, dt)

        if store_history:
            # We must .copy() the data, otherwise we just store
            # a reference to the same array that keeps changing.
            positions_history[step] = r_pos.copy()

        # Force computation: 
        # F_new = F^(t+1) at x^(t+1))
        F_new[:] = compute_nbody_force(r_pos, masses, device=device)
        
        # Velocity Integration
        # v^(t+1) = v^t + 0.5 * (a^t + a^(t+1)) * dt
        velocity_integration_kernel(v_vel, masses, F_old, F_new, dt)
        
        # Swap buffers
        F_old, F_new = F_new, F_old 
        
    print(f"Simulation finished.")

    if store_history:
        return positions_history
    else:
        return r_pos, v_vel
