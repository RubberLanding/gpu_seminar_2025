from nbody.pytorch.simulation import run_simulation_torch
from tests.nbody.test_helpers import check_kepler_orbit, check_energy_conservation
    
def test_kepler_orbit_torch():
    check_kepler_orbit(run_simulation_torch)
    print("Done.")

def test_energy_conservation_torch():
    check_energy_conservation(run_simulation_torch)
    print("Done.")