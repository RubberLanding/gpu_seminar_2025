import cupy as cp
import numpy as np
import math

# Constants (Must be passed to kernel or hardcoded)
G = 6.67430e-11
EPSILON = 1e-4

# 1. DEFINE THE C++ KERNEL FOR FORCES
# We use RawKernel for the heavy lifting to keep memory usage low (O(N)).
force_kernel_source = r'''
extern "C" __global__
void compute_forces(const double* pos, const double* masses, double* force, 
                    int N, double G, double EPSILON) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        double ftmp_x = 0.0;
        double ftmp_y = 0.0;
        double ftmp_z = 0.0;
        
        // Load i-th particle data
        double r_i_x = pos[i * 3 + 0];
        double r_i_y = pos[i * 3 + 1];
        double r_i_z = pos[i * 3 + 2];
        double m_i = masses[i];

        // Loop over all other particles
        for (int j = 0; j < N; j++) {
            double dx = r_i_x - pos[j * 3 + 0];
            double dy = r_i_y - pos[j * 3 + 1];
            double dz = r_i_z - pos[j * 3 + 2];

            double d2 = dx*dx + dy*dy + dz*dz;
            double dist = sqrt(d2 + EPSILON*EPSILON);
            
            double val = masses[j] / (dist * dist * dist);

            ftmp_x += dx * val;
            ftmp_y += dy * val;
            ftmp_z += dz * val;
        }

        force[i * 3 + 0] = -G * m_i * ftmp_x;
        force[i * 3 + 1] = -G * m_i * ftmp_y;
        force[i * 3 + 2] = -G * m_i * ftmp_z;
    }
}
'''

# Compile the kernel once
compute_forces_cupy = cp.RawKernel(force_kernel_source, 'compute_forces')

def run_simulation_cupy(pos_host, vel_host, mass_host, dt, steps, store_history=True):
    """
    Run the N-body simulation using CuPy.
    """
    N = pos_host.shape[0]
    
    # Allocate history buffers on CPU to store intermediate results
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float64)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float64)
        # Store initial state
        pos_history[0] = pos_host.copy()
        vel_history[0] = vel_host.copy()
    else:
        pos_history = None
        vel_history = None

    print(f"Running on GPU (CuPy). N={N}, Steps={steps}")

    # Move data to GPU using CuPy 
    pos_device = cp.array(pos_host)
    vel_device = cp.array(vel_host)
    mass_device = cp.array(mass_host)

    # Allocate force buffers on GPU
    force_device_old = cp.zeros((N, 3), dtype=cp.float64)
    force_device_new = cp.zeros((N, 3), dtype=cp.float64)

    # Pre-calculate constants
    # Reshape mass for broadcasting: (N,) -> (N, 1)
    inv_m = 1.0 / mass_device[:, None]
    dt2_half = 0.5 * dt * dt
    dt_half = 0.5 * dt

    # Grid configuration for the force kernel
    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block

    # Initial force calculation
    compute_forces_cupy(
        (blocks,), (threads_per_block,), 
        (pos_device, mass_device, force_device_old, N, G, EPSILON)
    )

    for step in range(steps):
        # Update position
        pos_device += (vel_device * dt) + (force_device_old * inv_m * dt2_half)

        # Update force 
        compute_forces_cupy(
            (blocks,), (threads_per_block,), 
            (pos_device, mass_device, force_device_new, N, G, EPSILON)
        )

        # Update velocity
        vel_device += (force_device_old + force_device_new) * inv_m * dt_half

        # Store history 
        if store_history:
            pos_history[step + 1] = pos_device.get() 
            vel_history[step + 1] = vel_device.get()


        # Swap references
        force_device_old, force_device_new = force_device_new, force_device_old

    # Return results
    if store_history:
        return pos_history, vel_history
    else:
        return pos_device.get(), vel_device.get()
    
if __name__ == "__main__":
    num_bodies = 2000
    pos = np.random.rand(num_bodies, 3).astype(np.float64) * 100.0
    vel = np.random.rand(num_bodies, 3).astype(np.float64) - 0.5
    mass = np.random.rand(num_bodies).astype(np.float64) * 1e4
    
    dt = 0.01
    steps = 100

    history_pos, history_vel = run_simulation_cupy(pos, vel, mass, dt, steps)
    
    print("Simulation step complete.")
