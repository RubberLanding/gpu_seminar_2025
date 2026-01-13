from nbody.numba_.simulation import run_simulation_numba
from tests.nbody.test_helpers import check_kepler_orbit, check_energy_conservation
    
def test_kepler_orbit_numba():
    check_kepler_orbit(run_simulation_numba)
    print("Done.")

def test_energy_conservation_numba():
    check_energy_conservation(run_simulation_numba)
    print("Done.")