import numpy as np 

def generate_cubic_coordinates(n_particles, spacing=1.0):
    lin_coords = np.linspace(0, n_particles - 1, n_particles, dtype=np.float32) * spacing
    X, Y, Z = np.meshgrid(lin_coords, lin_coords, lin_coords, indexing='ij')
    initial_positions = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
    
    return initial_positions

def generate_solar_system(n_bodies=100):
    """
    Generates a stable 'Solar System' like setup for showcasing.
    Body 0 is a massive 'Sun'. The rest are planets orbiting it.
    """
    pos = np.zeros((n_bodies, 3), dtype=np.float64)
    vel = np.zeros((n_bodies, 3), dtype=np.float64)
    mass = np.random.rand(n_bodies).astype(np.float64) * 1e10  # Asteroids
    
    # 1. Setup the Sun (Massive, at center)
    mass[0] = 1.0e20 
    pos[0] = [0, 0, 0]
    vel[0] = [0, 0, 0]
    
    # 2. Setup Planets/Asteroids
    for i in range(1, n_bodies):
        # Random distance from 50 to 200
        dist = 50 + np.random.rand() * 150
        
        # Random angle
        theta = np.random.rand() * 2 * np.pi
        
        # Position (Flat disk for nicer visualization)
        pos[i, 0] = dist * np.cos(theta)
        pos[i, 1] = dist * np.sin(theta)
        pos[i, 2] = (np.random.rand() - 0.5) * 5 # Small Z variation
        
        # Velocity for circular orbit: v = sqrt(GM / r)
        # Vector direction: tangent to the circle (-sin, cos)
        v_orb = np.sqrt(6.6743e-11 * mass[0] / dist)
        vel[i, 0] = -v_orb * np.sin(theta)
        vel[i, 1] = v_orb * np.cos(theta)
        vel[i, 2] = 0
        
        # Make a few heavy planets
        if i < 5:
            mass[i] *= 1000  # Gas giants
        
        # pos *= 1000
        # vel /= 10
            
    return pos, vel, mass
