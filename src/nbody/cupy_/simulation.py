import cupy as cp
import numpy as np
import math
import argparse

G = 6.67430e-11
EPS = 1e-4

force_kernel_naive = r'''
extern "C" __global__
void compute_forces_cupy_naive(const float* pos, const float* mass, float* force, 
                               int n, float G, float EPSILON) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // Hoist particle i's data into registers
    float px_i = pos[i*3 + 0];
    float py_i = pos[i*3 + 1];
    float pz_i = pos[i*3 + 2];
    float m_i = mass[i];

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    for (int j = 0; j < n; j++) {
        // Calculate difference
        float dx = px_i - pos[j*3 + 0];
        float dy = py_i - pos[j*3 + 1];
        float dz = pz_i - pos[j*3 + 2];

        // Distance squared + inverse square root
        float d2 = dx*dx + dy*dy + dz*dz + EPSILON*EPSILON;
        float inv_dist = rsqrtf(d2); // Built-in CUDA fast inverse square root
        
        float inv_dist3 = inv_dist * inv_dist * inv_dist;
        float s = mass[j] * inv_dist3;

        // Accumulate components
        fx += dx * s;
        fy += dy * s;
        fz += dz * s;
    }

    // Factor out loop invariants
    float factor = -G * m_i;
    force[i*3 + 0] = fx * factor;
    force[i*3 + 1] = fy * factor;
    force[i*3 + 2] = fz * factor;
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
compute_forces_cupy_naive = cp.RawKernel(force_kernel_naive, 'compute_forces_cupy_naive', options=('-use_fast_math',))
compute_forces_cupy_tiled = cp.RawKernel(force_kernel_tiled, 'compute_forces_cupy_tiled', options=('-use_fast_math',))

import torch 
from pykeops.torch import LazyTensor
def compute_forces_cupy_keops(grid, block, args):
    """
    Wrapper to make KeOps behave like a CuPy kernel in your loop.
    args order matches your kernel: (pos, mass, force_out, N, G, EPS)
    """
    pos_cupy, mass_cupy, force_out_cupy, N, G_val, EPS_val = args
    G_val = float(G_val)
    EPS_val = float(EPS_val)
    
    # 1. Zero-Copy Bridge: CuPy -> PyTorch
    # We use torch.as_tensor ensuring the underlying pointer is shared, not copied.
    pos_torch = torch.as_tensor(pos_cupy, device='cuda')
    mass_torch = torch.as_tensor(mass_cupy, device='cuda')
    
    # 2. Define Symbolic Variables
    # x_i: (N, 1, 3) - target particles
    # x_j: (1, N, 3) - source particles
    x_i = LazyTensor(pos_torch[:, None, :]) 
    x_j = LazyTensor(pos_torch[None, :, :])
    m_j = LazyTensor(mass_torch[None, :, None]) # (1, N, 1) masses
    
    # 3. Define Symbolic Formula (Gravity)
    # r_ij = x_j - x_i  (Using j-i convention to match your kernel's sign logic)
    # Your kernel logic: f += (pos[i] - pos[j]) * mass[j] / dist^3
    # Then final force = f * (-G * m_i) -> This flips the sign.
    # So effectively: Force_i = G * m_i * sum( m_j * (pos[j] - pos[i]) / dist^3 )
    
    diff = x_j - x_i
    sq_dist = (diff ** 2).sum(-1) + (EPS_val ** 2)
    inv_dist3 = sq_dist.rsqrt() ** 3
    
    # The term inside the sum: mass_j * (x_j - x_i) / dist^3
    force_term = m_j * diff * inv_dist3
    
    # 4. Perform Reduction
    # sum(dim=1) collapses the 'j' axis (N neighbors)
    acc_force_torch = force_term.sum(dim=1) 
    
    # 5. Apply constants (G * m_i)
    # We do this in PyTorch to keep it fast
    # Note: Your kernel multiplies by -G*m_i at the end because it computed (pos_i - pos_j).
    # Since we computed (pos_j - pos_i) inside KeOps, we multiply by positive G*m_i.
    final_force = acc_force_torch * (G_val * mass_torch[:, None])
    
    # 6. Write back to output array
    # Copy the result into the pre-allocated CuPy array
    # We have to copy here because 'force_out' is provided by the simulation loop
    # and 'final_force' is a new tensor created by KeOps.
    # However, this copy is Device-to-Device (extremely fast).
    cp.copyto(force_out_cupy, cp.asarray(final_force))

def run_simulation_cupy(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_cupy_keops, threads=128, store_history=False):
    # --- SETUP & TRANSFER ---
    print(f"Running on GPU (CuPy). N={pos_host.shape[0]}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")
    N    = pos_host.shape[0]
    pos  = cp.array(pos_host,  dtype=cp.float32)
    vel  = cp.array(vel_host,  dtype=cp.float32)
    mass = cp.array(mass_host, dtype=cp.float32)
    force_old = cp.zeros((N, 3), dtype=cp.float32)
    force_new = cp.zeros((N, 3), dtype=cp.float32)

    # --- CONSTANTS PREP ---
    dt_vec   = cp.float32(dt)
    dt2_half = 0.5 * dt_vec * dt_vec
    dt_half  = 0.5 * dt_vec
    inv_m    = 1.0 / mass[:, None]

    blocks  = (N + threads - 1) // threads
    grid_cfg, block_cfg = (blocks,), (threads,)
    
    # History buffer
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        pos_history[0], vel_history[0] = pos.get(), vel.get() # .get() required for numpy conversion
    else:
        pos_history, vel_history = None, None

    # Initial force calculation
    compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_old, np.int32(N), np.float32(G), np.float32(EPS)))

    # --- START SIMULATION ---
    for step in range(steps):
        
        # [Step A] Update Position
        pos += (vel * dt_vec) + (force_old * inv_m * dt2_half)
        
        # [Step B] Compute Forces
        compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_new, np.int32(N), np.float32(G), np.float32(EPS)))
        
        # [Step C] Update Velocity
        vel += (force_old + force_new) * inv_m * dt_half
        
        if store_history:
            pos_history[step + 1] = pos.get()
            vel_history[step + 1] = vel.get()

        # [Step D] Swap References
        force_old, force_new = force_new, force_old
    # ===================================================

    # --- FINALIZE ---
    if store_history:
        return pos_history, vel_history
    else:
        return pos.get(), vel.get()
    
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

    run_simulation_cupy(pos, vel, mass, args.dt, args.steps, store_history=False, compute_forces_func=compute_forces_cupy_keops)
    
    print("Simulation step complete.")