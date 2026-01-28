from nbody.triton_.simulation import run_simulation_triton
from tests.nbody.test_helpers import check_kepler_orbit, check_energy_conservation
    
def test_kepler_orbit_triton():
    check_kepler_orbit(run_simulation_triton)
    print("Done.")

def test_energy_conservation_triton():
    check_energy_conservation(run_simulation_triton)
    print("Done.")