import cupy as cp
import numpy as np
import math
import argparse

# Constants (Must be passed to kernel or hardcoded)
G = 6.67430e-11
EPSILON = 1e-4

# We use RawKernel for the heavy lifting to keep memory usage low (O(N)).
force_kernel_naive = r'''
extern "C" __global__
void compute_forces_cupy_naive(const float* pos, const float* masses, float* force, 
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

# CUDA with Shared Memory Tiling
force_kernel_tiled = r'''
extern "C" __global__
void compute_forces_cupy_tiled(const float* pos, const float* masses, float* force, 
                          int N, float G, float EPSILON) {
    
    // Shared memory: 128 particles per tile
    __shared__ float sh_pos[128 * 4]; 

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float f_x = 0.0f, f_y = 0.0f, f_z = 0.0f;
    
    float r_i_x = 0.0f, r_i_y = 0.0f, r_i_z = 0.0f, m_i = 0.0f;
    if (i < N) {
        r_i_x = pos[i * 3 + 0];
        r_i_y = pos[i * 3 + 1];
        r_i_z = pos[i * 3 + 2];
        m_i = masses[i];
    }

    for (int j_start = 0; j_start < N; j_start += 128) {
        // Cooperative load
        int j_load = j_start + threadIdx.x;
        if (j_load < N) {
            sh_pos[threadIdx.x * 4 + 0] = pos[j_load * 3 + 0];
            sh_pos[threadIdx.x * 4 + 1] = pos[j_load * 3 + 1];
            sh_pos[threadIdx.x * 4 + 2] = pos[j_load * 3 + 2];
            sh_pos[threadIdx.x * 4 + 3] = masses[j_load];
        } else {
            sh_pos[threadIdx.x * 4 + 3] = 0.0f; // Padding
        }
        __syncthreads();

        if (i < N) {
            int tile_limit = (N - j_start < 128) ? (N - j_start) : 128;
            for (int k = 0; k < tile_limit; k++) {
                int j_global = j_start + k;
                
                // Self-interaction guard
                if (i != j_global) {
                    float dx = sh_pos[k * 4 + 0] - r_i_x;
                    float dy = sh_pos[k * 4 + 1] - r_i_y;
                    float dz = sh_pos[k * 4 + 2] - r_i_z;

                    // NUMERICAL FIX: reciprocal square root instead of 1/(d^3)
                    float d2 = dx*dx + dy*dy + dz*dz + (EPSILON * EPSILON);
                    float inv_dist = rsqrtf(d2); 
                    float inv_dist3 = inv_dist * inv_dist * inv_dist;
                    
                    float s = sh_pos[k * 4 + 3] * inv_dist3;
                    f_x += dx * s;
                    f_y += dy * s;
                    f_z += dz * s;
                }
            }
        }
        __syncthreads();
    }

    if (i < N) {
        force[i * 3 + 0] = G * m_i * f_x;
        force[i * 3 + 1] = G * m_i * f_y;
        force[i * 3 + 2] = G * m_i * f_z;
    }
}
'''

# Compile the kernel once
compute_forces_cupy_naive = cp.RawKernel(force_kernel_naive, 'compute_forces_cupy_naive')
compute_forces_cupy_tiled = cp.RawKernel(force_kernel_tiled, 'compute_forces_cupy_tiled', options=('-use_fast_math',))

def run_simulation_cupy(pos_host, vel_host, mass_host, dt, steps, force_func=compute_forces_cupy_tiled, store_history=False):
    N = pos_host.shape[0]

    # Allocate history buffers on CPU to store intermediate results
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        # Store initial state
        pos_history[0] = pos_host.copy()
        vel_history[0] = vel_host.copy()

    # Move data to GPU
    pos_dev = cp.array(pos_host, dtype=cp.float32)
    vel_dev = cp.array(vel_host, dtype=cp.float32)
    mass_dev = cp.array(mass_host, dtype=cp.float32)
    force_old = cp.zeros((N, 3), dtype=cp.float32)
    force_new = cp.zeros((N, 3), dtype=cp.float32)

    # Grid config
    threads = 128
    blocks = (N + threads - 1) // threads
    
    # Pre-casting constants to np.float32 is MANDATORY for RawKernel stability
    G_f = np.float32(6.67430e-11)
    EPS_f = np.float32(1e-4)
    dt_f = np.float32(dt)

    # Initial force
    force_func((blocks,), (threads,), (pos_dev, mass_dev, force_old, np.int32(N), G_f, EPS_f))
    
    for step in range(steps):
        
        # Velocity-Verlet Step 1
        pos_dev += vel_dev * dt_f + 0.5 * force_old / mass_dev[:, None] * (dt_f**2)
        
        # New Force
        force_func((blocks,), (threads,), (pos_dev, mass_dev, force_new, np.int32(N), G_f, EPS_f))
        
        # Velocity-Verlet Step 2
        vel_dev += 0.5 * (force_old + force_new) / mass_dev[:, None] * dt_f

        # Store history 
        if store_history:
            pos_history[step + 1] = pos_dev.get() 
            vel_history[step + 1] = vel_dev.get()

        # Swap
        force_old, force_new = force_new, force_old

    if store_history:
        return pos_history, vel_history
    else:
        return pos_dev.get(), vel_dev.get()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cupy N-Body Simulation")
    parser.add_argument("-n", "--num-bodies", type=int, default=1000, help="Number of particles")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    args = parser.parse_args()

    pos = np.random.rand(args.num_bodies, 3).astype(np.float32) * 100.0
    vel = np.random.rand(args.num_bodies, 3).astype(np.float32) - 0.5
    mass = np.random.rand(args.num_bodies).astype(np.float32) * 1e4
    
    print(f"Simulation with Cupy. Initializing {args.num_bodies} bodies...")

    run_simulation_cupy(pos, vel, mass, args.dt, args.steps, store_history=False)
    
    print("Simulation step complete.")