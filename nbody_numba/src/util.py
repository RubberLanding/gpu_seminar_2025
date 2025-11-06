import numpy as np 

def generate_cubic_coordinates(n_particles, spacing=1.0):
    lin_coords = np.linspace(0, n_particles - 1, n_particles, dtype=np.float64) * spacing
    X, Y, Z = np.meshgrid(lin_coords, lin_coords, lin_coords, indexing='ij')
    initial_positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    
    return initial_positions