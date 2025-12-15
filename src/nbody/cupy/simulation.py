import cupy as cp
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

        // Loop over all other particles (J)
        for (int j = 0; j < N; j++) {
            double dx = r_i_x - pos[j * 3 + 0];
            double dy = r_i_y - pos[j * 3 + 1];
            double dz = r_i_z - pos[j * 3 + 2];

            double d2 = dx*dx + dy*dy + dz*dz;
            double dist = sqrt(d2 + EPSILON*EPSILON);
            
            // Avoid division by zero if i == j (dist > epsilon handles this, but logic holds)
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
compute_forces_gpu = cp.RawKernel(force_kernel_source, 'compute_forces')

def step_simulation_cupy(r_pos, v_vel, masses, r_force, dt, N):
    """
    Main simulation step using CuPy.
    """
    # Grid configuration
    threads_per_block = 128
    blocks = (N + threads_per_block - 1) // threads_per_block

    # --- STEP 1: FORCE CALCULATION (Using RawKernel) ---
    # We pass the arrays directly. CuPy handles the pointers.
    compute_forces_gpu(
        (blocks,), (threads_per_block,), 
        (r_pos, masses, r_force, N, G, EPSILON)
    )

    # --- STEP 2: POSITION UPDATE (Using Vectorized Operations) ---
    # In CuPy, we replace the 'gpu_step_pos' kernel with array math.
    # Note: masses[:, None] reshapes masses from (N,) to (N, 1) for broadcasting
    inv_m = 1.0 / masses[:, None] 
    dt2_half = 0.5 * dt * dt
    
    # Store F_old for the next velocity step (Verlet integration)
    F_old = r_force.copy() 
    
    # Update Position: r = r + v*dt + a*0.5*dt^2
    r_pos += (v_vel * dt) + (F_old * inv_m * dt2_half)

    # --- STEP 3: RE-CALCULATE FORCES (New Position) ---
    # Velocity Verlet requires force at t+1 to update velocity
    F_new = cp.zeros_like(r_force)
    compute_forces_gpu(
        (blocks,), (threads_per_block,), 
        (r_pos, masses, F_new, N, G, EPSILON)
    )

    # --- STEP 4: VELOCITY UPDATE (Using Vectorized Operations) ---
    # v = v + 0.5 * (a_old + a_new) * dt
    dt_half = 0.5 * dt
    v_vel += (F_old + F_new) * inv_m * dt_half
    
    # Update the main force array for the next iteration
    r_force[:] = F_new

    return r_pos, v_vel, r_force

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    N_PARTICLES = 1024
    
    # Initialize data on GPU directly using CuPy
    pos = cp.random.rand(N_PARTICLES, 3).astype(cp.float64)
    vel = cp.random.rand(N_PARTICLES, 3).astype(cp.float64)
    mass = cp.ones(N_PARTICLES, dtype=cp.float64)
    force = cp.zeros((N_PARTICLES, 3), dtype=cp.float64)
    
    dt = 0.01

    # Run one step
    pos, vel, force = step_simulation_cupy(pos, vel, mass, force, dt, N_PARTICLES)
    
    print("Simulation step complete.")
    print("New Position [0]:", pos[0])