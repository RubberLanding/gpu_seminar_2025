# If you run the file directly from the terminal, 
# use `PYTHONPATH=src python -s test/test.py` for the imports to work

import numpy as np
import math
import pytest
from numba import njit

# Adjust imports based on your exact file structure
# If using PYTHONPATH=src, these work:
from simulation import run_simulation
from visualization import generate_solar_system

@njit
def calculate_energy(pos, vel, masses, G, epsilon):
    """Calculates Total Energy (Kinetic + Potential)"""
    n = len(masses)
    kinetic = 0.0
    potential = 0.0
    
    # Kinetic Energy
    for i in range(n):
        v2 = vel[i, 0]**2 + vel[i, 1]**2 + vel[i, 2]**2
        kinetic += 0.5 * masses[i] * v2

    # Potential Energy
    for i in range(n):
        for j in range(i + 1, n): 
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz + epsilon**2)
            potential -= (G * masses[i] * masses[j]) / dist
            
    return kinetic + potential

def analyze_energy(pos_history, vel_history, masses, G, epsilon):
    steps = pos_history.shape[0]
    energy_history = np.zeros(steps)
    
    print("Computing energy profile...")
    for i in range(steps):
        energy_history[i] = calculate_energy(pos_history[i], vel_history[i], masses, G, epsilon)
        
    return energy_history

def test_kepler_orbit_accuracy():
    """
    Verifies that a 2-body system follows a stable circular orbit 
    and returns to the start after exactly one period.
    """
    
    G_test = 6.6743e-11
    M_sun = 1.0e20
    R = 1.0e6  # 1,000 km orbit
    
    masses = np.array([M_sun, 10.0], dtype=np.float64)
    pos = np.zeros((2, 3), dtype=np.float64)
    vel = np.zeros((2, 3), dtype=np.float64)
    
    # Place Earth at (R, 0, 0)
    pos[1, 0] = R 
    
    # Calculate Velocity for perfect circle: v = sqrt(GM / R)
    v_circ = math.sqrt(G_test * M_sun / R)
    vel[1, 1] = v_circ
    
    # Calculate exact Orbital Period: T = 2*pi*R / v
    period = 2 * math.pi * R / v_circ
    
    # Velocity is approx 81 m/s. 
    # With dt=100s, it moves 8km per step. 8km is small compared to R=1000km.
    dt = 10.0 
    steps = int(period / dt)
    
    print(f"Testing Kepler: R={R:.0f}, v={v_circ:.2f}, Period={period:.2f}s, Steps={steps}")

    pos_hist, _ = run_simulation(pos, vel, masses, dt=dt, steps=steps, device="auto")
    
    # Compute the distance of Earth relative to the Sun
    rel_pos = pos_hist[:, 1, :] - pos_hist[:, 0, :]
    distances = np.sqrt(np.sum(rel_pos**2, axis=1))
    
    # Calculate deviation as a percentage of Radius
    max_diff = np.max(np.abs(distances - R))
    percent_error = (max_diff / R) * 100
    
    print(f"Max Radius Deviation: {max_diff:.2f} meters ({percent_error:.4f}%)")
    
    # Allow 0.1% deviation (Energy drift in symplectic integrators is normal)
    assert percent_error < 0.1, f"Orbit not circular. Error: {percent_error:.4f}%"

    # B. Check Periodicity (Final position ~= Initial position)
    final_pos = rel_pos[-1]
    
    # Check X coordinate (Should be back at R)
    assert final_pos[0] == pytest.approx(R, rel=0.01)
    
    # Check Y coordinate (Should be back at 0)
    assert final_pos[1] == pytest.approx(0.0, abs=R * 0.05) 

def test_energy_conservation():
    """
    Test that the energy is conserved.
    """
    N = 10
    pos, vel, mass = generate_solar_system(N)
    
    n_steps = 400
    pos_hist, vel_hist = run_simulation(pos, vel, mass, dt=0.001, steps=n_steps, device="auto")
    energies = analyze_energy(pos_hist, vel_hist, mass, G=6.6743e-11, epsilon=1e-1)

    # Calculate drift
    drift = (np.max(energies) - np.min(energies)) / np.mean(np.abs(energies[1:n_steps])) 
    print(f"Energy Drift: {drift * 100:.6f}%")

    # # Check visually
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.plot(energies)
    # plt.title("Total Energy over Time")
    # plt.xlabel("Step")
    # plt.ylabel("Energy (Joules)")
    # plt.grid(True)
    # plt.show()
    
    # Drift should be less than 1% for this setup
    assert drift < 0.01
    
if __name__ == "__main__":
    test_kepler_orbit_accuracy()
    test_energy_conservation()
    print("Done.")