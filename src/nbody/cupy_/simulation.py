import cupy as cp
import numpy as np
import math
import argparse

# Constants (Must be passed to kernel or hardcoded)
G = 6.67430e-11
EPSILON = 1e-4

# 1. DEFINE THE C++ KERNEL FOR FORCES
# We use RawKernel for the heavy lifting to keep memory usage low (O(N)).
force_kernel_source = r'''
extern "C" __global__
void compute_forces(const float* pos, const float* masses, float* force, 
                    int N, float G, float EPSILON) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float ftmp_x = 0.0f;
        float ftmp_y = 0.0f;
        float ftmp_z = 0.0f;
        
        float r_i_x = pos[i * 3 + 0];
        float r_i_y = pos[i * 3 + 1];
        float r_i_z = pos[i * 3 + 2];
        float m_i = masses[i];

        for (int j = 0; j < N; j++) {
            float dx = r_i_x - pos[j * 3 + 0];
            float dy = r_i_y - pos[j * 3 + 1];
            float dz = r_i_z - pos[j * 3 + 2];

            float d2 = dx*dx + dy*dy + dz*dz;
            float dist = sqrtf(d2 + EPSILON*EPSILON); // Note: sqrtf for float
            
            float val = masses[j] / (dist * dist * dist);

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

def run_simulation_cupy(pos_host, vel_host, mass_host, dt, steps, store_history=False):
    """
    Run the N-body simulation using CuPy.
    """
    N = pos_host.shape[0]
    
    # Allocate history buffers on CPU to store intermediate results
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
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
    force_device_old = cp.zeros((N, 3), dtype=cp.float32)
    force_device_new = cp.zeros((N, 3), dtype=cp.float32)

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
    parser = argparse.ArgumentParser(description="Cupy N-Body Simulation")
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of particles")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    args = parser.parse_args()

    pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
    vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
    mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4
    
    print(f"Simulation with Numba. Initializing {args.num_bodies} bodies...")

    run_simulation_cupy(pos, vel, mass, args.dt, args.steps, store_history=False)
    
    print("Simulation step complete.")