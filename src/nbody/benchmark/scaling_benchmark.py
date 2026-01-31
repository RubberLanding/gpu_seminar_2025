import inspect
import argparse
import numpy as np

from nbody.benchmark.benchmark import measure_time_cupy, measure_time_numba, measure_time_torch, measure_time_triton
from nbody.benchmark.util import cleanup_gpu, store_results, plot_results, create_report
from nbody.pytorch_.simulation import compute_forces_pytorch_naive, compute_forces_pytorch_chunked, compute_forces_pytorch_keops, compute_forces_pytorch_matmul, compute_forces_pytorch_optimized
from nbody.cupy_.simulation import compute_forces_cupy_naive, compute_forces_cupy_tiled
from nbody.numba_.simulation import compute_forces_numba_naive, compute_forces_numba_tiled
from nbody.triton_.simulation import compute_forces_triton_naive

def run_scaling_benchmark(measure_time_func, n_particles, compute_forces=None, **kwargs):
    results = {
        "num_bodies": [],
        "total_time": [],
        "steps_per_second": [],
        "interactions_per_second": []}
    max_wait_time = 180.0 # Don't run a size if 1 step takes more than max_wait_time

    for n in n_particles:
        np.random.seed(42) 
        pos = np.random.rand(n, 3).astype(np.float32) * 100.0
        vel = np.random.rand(n, 3).astype(np.float32) - 0.5
        mass = np.random.rand(n).astype(np.float32) * 1e4

        if compute_forces is not None: 
            res = measure_time_func(pos, vel, mass, compute_forces_func=compute_forces, **kwargs)
        else:
            res = measure_time_func(pos, vel, mass, **kwargs)

        steps, total_time, steps_per_second, interactions_per_second = res
        results["num_bodies"].append(n)
        results["total_time"].append(total_time)
        results["steps_per_second"].append(steps_per_second)
        results["interactions_per_second"].append(interactions_per_second)

        # If the current time a step takes is too long already, do not run
        # any additional benchmarks with even larger problem size. 
        time_per_step = total_time/steps
        if time_per_step > max_wait_time:
            print(f"Step time is on average {time_per_step:.2f}, which is longer than the maximum waiting time {max_wait_time:.2f}.\n")
            print("Ending the scaling benchmark now.")
            break
        
    return results

if __name__== "__main__":
    parser = argparse.ArgumentParser(description="N-Body Simulation Benchmark")
    parser.add_argument("--n-start", type=int, default=8, help="The number of particles are calculated like: `n_i = (4 * i)^3 for i in (n_start, ..., n_end)`.")
    parser.add_argument("--n-end", type=int, default=8, help="The number of particles are calculated like: `n_i = (4 * i)^3 for i in (n_start, ..., n_end)`.")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("-m", "--method", type=str, choices=["numba", "torch", "cupy", "triton", "all"], default="numba", help="Method to use (numba, torch, cupy, triton or all)")
    parser.add_argument("-f", "--force-func", type=str, choices=["compute_forces_triton_naive", 
                                                                 "compute_forces_cupy_naive", "compute_forces_cupy_tiled",
                                                                 "compute_forces_numba_naive", "compute_forces_numba_tiled",
                                                                 "compute_forces_pytorch_naive", "compute_forces_pytorch_chunked", "compute_forces_pytorch_keops", 
                                                                 "compute_forces_pytorch_matmul", "compute_forces_pytorch_optimized"], default="default",
                                                                 help="The force function that is being used, e.g. `gpu_force_kernel_numba_naive` for Numba.")
    parser.add_argument("--store-results", action="store_true", help="Store the results.") 
    parser.add_argument("--store-plot", action="store_true", help="Store the performance plot.") 
    args = parser.parse_args()

    # Mapping of framework name to its measure function and allowed force kernels
    FRAMEWORK_CONFIG = {
        "cupy": {
            "measure": measure_time_cupy,
            "kernels": {
                "compute_forces_cupy_naive": compute_forces_cupy_naive,
                "compute_forces_cupy_tiled": compute_forces_cupy_tiled,
                "default":                   inspect.signature(measure_time_cupy).parameters["compute_forces_func"].default
            }
        },
        "numba": {
            "measure": measure_time_numba,
            "kernels": {
                "compute_forces_numba_naive": compute_forces_numba_naive,
                "compute_forces_numba_tiled": compute_forces_numba_tiled,
                "default":                    inspect.signature(measure_time_numba).parameters["compute_forces_func"].default
            }
        },
        "triton": {
            "measure": measure_time_triton,
            "kernels": {
                "compute_forces_triton_naive": compute_forces_triton_naive,
                "default":                     inspect.signature(measure_time_triton).parameters["compute_forces_func"].default
            }
        },
        "torch": {
            "measure": measure_time_torch,
            "kernels": {
                "compute_forces_pytorch_naive":     compute_forces_pytorch_naive,
                "compute_forces_pytorch_chunked":   compute_forces_pytorch_chunked,
                "compute_forces_pytorch_keops":     compute_forces_pytorch_keops,
                "compute_forces_pytorch_matmul":    compute_forces_pytorch_matmul,
                "compute_forces_pytorch_optimized": compute_forces_pytorch_optimized,
                "default":                          inspect.signature(measure_time_torch).parameters["compute_forces_func"].default
            }
        }
    }

    print("START SCALING BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")

    n_particles = [(4 * i)**3 for i in range(args.n_start, args.n_end + 1)]
    methods_to_run = FRAMEWORK_CONFIG.keys() if args.method == "all" else [args.method]

    for method in methods_to_run:
        config = FRAMEWORK_CONFIG[method]

        print(f"Measure {method.capitalize()}...")
        print(f"Force function is {args.force_func}")

        if args.force_func in config["kernels"]:
            force_func = config["kernels"][args.force_func]
        else:
            # If a specific force-func was requested but doesn't belong to this method
            print(f"Skipping {method}: '{args.force_func}' is incompatible.")
            print("-" * 20)
            continue

        results = run_scaling_benchmark(
            config["measure"], 
            n_particles, 
            compute_forces=force_func, 
            dt=args.dt, 
            steps=args.steps
        )

        force_func_str = force_func.__name__

        if args.store_results or args.store_plot: report_folder, timestamp = create_report(force_func_str)
        if args.store_results: store_results(force_func_str, results, timestamp, report_folder)
        if args.store_plot: plot_results(force_func_str, results["num_bodies"], results["interactions_per_second"], report_folder)

        cleanup_gpu()
        print("-" * 20 + "\n")

    print("END SCALING BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")