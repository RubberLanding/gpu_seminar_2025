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

force_kernel_tiled = r'''
extern "C" __global__
void compute_forces_cupy_tiled(const float* pos, const float* mass, float* force, 
                                   int n, float G, float EPSILON) {
    // Allocate shared memory for the tile (supports up to 1024 threads per block)
    __shared__ float sh_px[1024];
    __shared__ float sh_py[1024];
    __shared__ float sh_pz[1024];
    __shared__ float sh_m[1024];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Hoist particle i's data into registers
    float px_i = 0.0f, py_i = 0.0f, pz_i = 0.0f, m_i = 0.0f;
    if (i < n) {
        px_i = pos[i*3 + 0];
        py_i = pos[i*3 + 1];
        pz_i = pos[i*3 + 2];
        m_i  = mass[i];
    }

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    // Calculate how many tiles we need to cover all 'n' particles
    int num_tiles = (n + blockDim.x - 1) / blockDim.x;

    for (int tile = 0; tile < num_tiles; tile++) {
        // 1. Every thread cooperatively loads ONE particle into shared memory
        int j = tile * blockDim.x + threadIdx.x;
        
        if (j < n) {
            sh_px[tid] = pos[j*3 + 0];
            sh_py[tid] = pos[j*3 + 1];
            sh_pz[tid] = pos[j*3 + 2];
            sh_m[tid]  = mass[j];
        } else {
            // Pad out-of-bounds with zero mass to nullify force contributions
            sh_px[tid] = 0.0f;
            sh_py[tid] = 0.0f;
            sh_pz[tid] = 0.0f;
            sh_m[tid]  = 0.0f; 
        }
        
        // Ensure all threads have finished writing to shared memory
        __syncthreads();

        // 2. Compute forces against all particles in the current tile
        #pragma unroll 16
        for (int k = 0; k < blockDim.x; k++) {
            float dx = px_i - sh_px[k];
            float dy = py_i - sh_py[k];
            float dz = pz_i - sh_pz[k];

            float d2 = dx*dx + dy*dy + dz*dz + EPSILON*EPSILON;
            float inv_dist = rsqrtf(d2);
            float inv_dist3 = inv_dist * inv_dist * inv_dist;
            
            float s = sh_m[k] * inv_dist3;

            fx += dx * s;
            fy += dy * s;
            fz += dz * s;
        }
        
        // Ensure all threads are done reading before the next tile overwrites shared memory
        __syncthreads();
    }

    // Write final force back to global memory
    if (i < n) {
        float factor = -G * m_i;
        force[i*3 + 0] = fx * factor;
        force[i*3 + 1] = fy * factor;
        force[i*3 + 2] = fz * factor;
    }
}
'''

# Compile the kernel once
compute_forces_cupy_naive = cp.RawKernel(force_kernel_naive, 'compute_forces_cupy_naive', options=('-use_fast_math', '-lineinfo', '-std=c++17'), backend='nvcc')
compute_forces_cupy_tiled = cp.RawKernel(force_kernel_tiled, 'compute_forces_cupy_tiled', options=('-use_fast_math', '-lineinfo', '-std=c++17'), backend='nvcc')
compute_forces_cupy_optimized = None

def run_simulation_cupy(pos_host, vel_host, mass_host, dt, steps, compute_forces_func=compute_forces_cupy_naive, threads=128, store_history=False):
    # [FIX] Move kernel compilation inside the function and use an f-string 
    # to statically allocate shared memory using the `threads` variable!
    if compute_forces_func is None:
        force_kernel_optimized = f'''
        extern "C" __global__
        void compute_forces_cupy_optimized(
            const float* __restrict__ pos_x,
            const float* __restrict__ pos_y,
            const float* __restrict__ pos_z,
            const float* __restrict__ mass,
            float* __restrict__ force_x,
            float* __restrict__ force_y,
            float* __restrict__ force_z,
            int N, float G, float EPSILON)
        {{
            // STATICALLY allocated shared memory (Bypasses the cupy kwarg bug)
            __shared__ float sh_pos_x[{threads}];
            __shared__ float sh_pos_y[{threads}];
            __shared__ float sh_pos_z[{threads}];
            __shared__ float sh_mass[{threads}];

            int i = blockIdx.x * blockDim.x + threadIdx.x;
            
            float rx = 0.0f, ry = 0.0f, rz = 0.0f, m_i = 0.0f;
            if (i < N) {{
                rx = pos_x[i];
                ry = pos_y[i];
                rz = pos_z[i];
                m_i = mass[i];
            }}

            float fx = 0.0f;
            float fy = 0.0f;
            float fz = 0.0f;
            float eps2 = EPSILON * EPSILON;

            int num_tiles = (N + blockDim.x - 1) / blockDim.x;

            for (int tile = 0; tile < num_tiles; tile++) {{
                // 1. Load data into shared memory
                int load_idx = tile * blockDim.x + threadIdx.x;
                
                if (load_idx < N) {{
                    sh_pos_x[threadIdx.x] = pos_x[load_idx];
                    sh_pos_y[threadIdx.x] = pos_y[load_idx];
                    sh_pos_z[threadIdx.x] = pos_z[load_idx];
                    sh_mass[threadIdx.x]  = mass[load_idx];
                }} else {{
                    sh_pos_x[threadIdx.x] = 0.0f;
                    sh_pos_y[threadIdx.x] = 0.0f;
                    sh_pos_z[threadIdx.x] = 0.0f;
                    sh_mass[threadIdx.x]  = 0.0f; 
                }}

                __syncthreads();

                // 2. Compute forces
                if (i < N) {{
                    #pragma unroll 8
                    for (int j = 0; j < {threads}; j++) {{
                        float dx = sh_pos_x[j] - rx;
                        float dy = sh_pos_y[j] - ry;
                        float dz = sh_pos_z[j] - rz;

                        float d2 = dx*dx + dy*dy + dz*dz + eps2;
                        float inv_dist = rsqrtf(d2); 
                        float s = sh_mass[j] * inv_dist * inv_dist * inv_dist;

                        fx += dx * s;
                        fy += dy * s;
                        fz += dz * s;
                    }}
                }}
                __syncthreads();
            }}

            // Write coalesced forces back
            if (i < N) {{
                force_x[i] = fx * G * m_i;
                force_y[i] = fy * G * m_i;
                force_z[i] = fz * G * m_i;
            }}
        }}
        '''
        
        # Compile the dynamically generated kernel (CuPy heavily caches this anyway)
        compute_forces_func = cp.RawKernel(
            force_kernel_optimized, 
            'compute_forces_cupy_optimized', 
            options=('-use_fast_math', '-lineinfo',  '-std=c++17'), 
            backend='nvcc'
        )

    # --- SETUP & TRANSFER ---
    N = pos_host.shape[0]
    blocks = (N + threads - 1) // threads

    print(f"Running on GPU (CuPy). N={N}, Steps={steps}")
    print(f"Using Force Function: {compute_forces_func.__name__}")
    
    func_name = getattr(compute_forces_func, '__name__', 'Unknown')
    is_soa = ("optimized" in func_name) 

    # History buffer
    if store_history:
        pos_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        vel_history = np.zeros((steps + 1, N, 3), dtype=np.float32)
        pos_history[0], vel_history[0] = pos_host.copy(), vel_host.copy() 
    else:
        pos_history, vel_history = None, None

    if is_soa:
        # =========================================================
        # STRUCTURE OF ARRAYS (e.g. optimized kernel)
        # =========================================================
        px = np.ascontiguousarray(pos_host[:, 0], dtype=np.float32)
        py = np.ascontiguousarray(pos_host[:, 1], dtype=np.float32)
        pz = np.ascontiguousarray(pos_host[:, 2], dtype=np.float32)
        
        vx = np.ascontiguousarray(vel_host[:, 0], dtype=np.float32)
        vy = np.ascontiguousarray(vel_host[:, 1], dtype=np.float32)
        vz = np.ascontiguousarray(vel_host[:, 2], dtype=np.float32)
        
        mass_arr = np.ascontiguousarray(mass_host, dtype=np.float32)
        inv_mass_arr = np.ascontiguousarray(1.0 / mass_host, dtype=np.float32)

        # Transfer to GPU
        d_px, d_py, d_pz = cp.asarray(px), cp.asarray(py), cp.asarray(pz)
        d_vx, d_vy, d_vz = cp.asarray(vx), cp.asarray(vy), cp.asarray(vz)
        d_mass, d_inv_mass = cp.asarray(mass_arr), cp.asarray(inv_mass_arr)

        d_f_old_x = cp.empty(N, dtype=cp.float32)
        d_f_old_y = cp.empty(N, dtype=cp.float32)
        d_f_old_z = cp.empty(N, dtype=cp.float32)
        
        d_f_new_x = cp.empty(N, dtype=cp.float32)
        d_f_new_y = cp.empty(N, dtype=cp.float32)
        d_f_new_z = cp.empty(N, dtype=cp.float32)

        G = np.float32(6.6743e-11)
        EPSILON = np.float32(1e-5)
        
        compute_forces_func(
                (blocks,), (threads,), 
                (d_px, d_py, d_pz, d_mass, d_f_new_x, d_f_new_y, d_f_new_z, np.int32(N), np.float32(G), np.float32(EPSILON))
            )        
        
        cp.cuda.Stream.null.synchronize()

        dt_fp32 = cp.float32(dt)
        half_dt_fp32 = cp.float32(0.5 * dt)
        half_dt2_fp32 = cp.float32(0.5 * dt * dt)

        # --- MAIN LOOP ---
        for step in range(steps):
            # Step A: Update Position (FIXED)
            d_px += (d_vx * dt_fp32) + (d_f_old_x * d_inv_mass * half_dt2_fp32)
            d_py += (d_vy * dt_fp32) + (d_f_old_y * d_inv_mass * half_dt2_fp32)
            d_pz += (d_vz * dt_fp32) + (d_f_old_z * d_inv_mass * half_dt2_fp32)

            # Step B: Compute Forces
            compute_forces_func(
                (blocks,), (threads,), 
                (d_px, d_py, d_pz, d_mass, d_f_new_x, d_f_new_y, d_f_new_z, np.int32(N), np.float32(G), np.float32(EPSILON))
            )        

            # Step C: Update Velocity
            d_vx += (d_f_old_x + d_f_new_x) * d_inv_mass * half_dt_fp32
            d_vy += (d_f_old_y + d_f_new_y) * d_inv_mass * half_dt_fp32
            d_vz += (d_f_old_z + d_f_new_z) * d_inv_mass * half_dt_fp32

            if store_history:
                pos_history[step + 1, :, 0] = d_px.get()
                pos_history[step + 1, :, 1] = d_py.get()
                pos_history[step + 1, :, 2] = d_pz.get()

                vel_history[step + 1, :, 0] = d_vx.get()
                vel_history[step + 1, :, 1] = d_vy.get()
                vel_history[step + 1, :, 2] = d_vz.get()

            # Step D: Pointer Swap
            d_f_old_x, d_f_new_x = d_f_new_x, d_f_old_x
            d_f_old_y, d_f_new_y = d_f_new_y, d_f_old_y
            d_f_old_z, d_f_new_z = d_f_new_z, d_f_old_z

        # --- FINALIZE ---
        if store_history:
            return pos_history, vel_history
        else:
            pos = np.zeros((N, 3), dtype=np.float32)
            vel = np.zeros((N, 3), dtype=np.float32)
            
            # FIX 3: Use .get()
            pos[:, 0] = d_px.get()
            pos[:, 1] = d_py.get()
            pos[:, 2] = d_pz.get()

            vel[:, 0] = d_vx.get()
            vel[:, 1] = d_vy.get()
            vel[:, 2] = d_vz.get()

            return pos, vel

    else:
        # =========================================================
        # ARRAY OF STRUCTURES (e.g. naive kernel)
        # =========================================================
        pos  = cp.array(pos_host,  dtype=cp.float32)
        vel  = cp.array(vel_host,  dtype=cp.float32)
        mass = cp.array(mass_host, dtype=cp.float32)
        force_old = cp.zeros((N, 3), dtype=cp.float32)
        force_new = cp.zeros((N, 3), dtype=cp.float32)

        G_val = np.float32(6.6743e-11)
        EPSILON_val = np.float32(1e-5)

        # --- CONSTANTS PREP ---
        dt_vec   = cp.float32(dt)
        dt2_half = 0.5 * dt_vec * dt_vec
        dt_half  = 0.5 * dt_vec
        inv_m    = 1.0 / mass[:, None]

        grid_cfg, block_cfg = (blocks,), (threads,)

        # Initial force calculation
        compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_old, np.int32(N), G_val, EPSILON_val))

        # --- START SIMULATION ---
        for step in range(steps):
            
            # [Step A] Update Position
            pos += (vel * dt_vec) + (force_old * inv_m * dt2_half)
            
            # [Step B] Compute Forces
            compute_forces_func(grid_cfg, block_cfg, (pos, mass, force_new, np.int32(N), G_val, EPSILON_val))
            
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

    run_simulation_cupy(pos, vel, mass, args.dt, args.steps, store_history=False, compute_forces_func=None)
    
    print("Simulation step complete.")