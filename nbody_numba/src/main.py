import numpy as np

from simulation import run_rk2_simulation
from visualization import animate_simulation, plot_snapshot
from util import generate_cubic_coordinates

if __name__ == '__main__':
    length = 4
    mass = 1.5e10
    delta_t = 5e-5        
    n_iteration = 10000   
    
    # Use float64 as required by Numba/CUDA signatures
    np.random.seed(42)
    initial_positions = generate_cubic_coordinates(length)
    initial_velocities = np.zeros((length**3, 3), dtype=np.float64)
    particle_masses = np.full(length**3, mass, dtype=np.float64)

    # Set device="auto" to use GPU if available, or "cpu" to force CPU.
    positions_history = run_rk2_simulation(
        initial_positions, 
        initial_velocities, 
        particle_masses, 
        dt=delta_t, 
        n_loops=n_iteration,
        device="cpu"
    )
    animate_simulation(positions_history)
    # plot_snapshot(positions=positions_history[0])    
    # plot_snapshot(positions=positions_history[n_iteration-1])

    i=0


