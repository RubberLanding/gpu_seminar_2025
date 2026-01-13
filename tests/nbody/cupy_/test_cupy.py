from nbody.cupy_.simulation import run_simulation_cupy
from tests.nbody.test_helpers import check_kepler_orbit, check_energy_conservation
    
def test_kepler_orbit_cupy():
    check_kepler_orbit(run_simulation_cupy)
    print("Done.")

def test_energy_conservation_cupy():
    check_energy_conservation(run_simulation_cupy)
    print("Done.")