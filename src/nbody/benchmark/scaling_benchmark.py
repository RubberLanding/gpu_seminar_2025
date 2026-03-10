import inspect
import argparse
import numpy as np

from nbody.benchmark.benchmark import measure_time_cupy, measure_time_numba, measure_time_torch, measure_time_triton
from nbody.benchmark.util import cleanup_gpu, store_results, plot_results, create_report
from nbody.pytorch_.simulation import compute_forces_pytorch_naive, compute_forces_pytorch_keops 
from nbody.cupy_.simulation import compute_forces_cupy_naive, compute_forces_cupy_optimized
from nbody.numba_.simulation import compute_forces_numba_naive, compute_forces_numba_optimized
from nbody.triton_.simulation import compute_accel_triton_naive, compute_accel_triton_optimized 

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
    parser.add_argument("-ns", "--n-start", type=int, default=8, help="The number of particles are calculated like: `n_i = (4 * i)^3 for i in (n_start, ..., n_end)`.")
    parser.add_argument("-ne", "--n-end", type=int, default=19, help="The number of particles are calculated like: `n_i = (4 * i)^3 for i in (n_start, ..., n_end)`.")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Number of steps per run")
    parser.add_argument("-dt", "--dt", type=float, default=0.01, help="Time step size")
    parser.add_argument("-f", "--force-func", type=str, nargs="+", choices=[
            "compute_accel_triton_naive", "compute_accel_triton_optimized", "compute_accel_triton_tensor", "compute_accel_triton_tiled", "compute_accel_triton_mixed",
            "compute_forces_cupy_naive", "compute_forces_cupy_tiled", "compute_forces_cupy_keops", "compute_forces_cupy_optimized",
            "compute_forces_numba_naive", "compute_forces_numba_tiled", "compute_forces_numba_optimized", 
            "compute_forces_pytorch_naive", "compute_forces_pytorch_chunked", "compute_forces_pytorch_keops", 
            "compute_forces_pytorch_matmul", "compute_forces_pytorch_optimized"], 
            help="One or more force functions to benchmark.")
    parser.add_argument("-t", "--threads", type=int, default=128, help="Threads per block for Numba and Cupy. Should be a multiple of 32.")
    parser.add_argument("-bt", "--bs-triton", type=int, default=32, help="Block size for Triton. Should be a multiple of 16.")
    parser.add_argument("-sr", "--store-results", action="store_true", help="Store the results.")
    parser.add_argument("-sp", "--store-plot", action="store_true", help="Store the performance plot.") 
    args = parser.parse_args()

    assert args.force_func != None, "Provide a force function, e.g. `--force-func compute_forces_cupy_naive`!"

    # Mapping of framework name to its measure function and allowed force kernels
    FRAMEWORK_CONFIG = {
        "cupy": {
            "measure": measure_time_cupy,
            "kernels": {
                "compute_forces_cupy_naive": compute_forces_cupy_naive,
                "compute_forces_cupy_optimized": compute_forces_cupy_optimized,
                # "compute_forces_cupy_tiled": compute_forces_cupy_tiled,
                # "compute_forces_cupy_keops": compute_forces_cupy_keops,
            }
        },
        "numba": {
            "measure": measure_time_numba,
            "kernels": {
                "compute_forces_numba_naive": compute_forces_numba_naive,
                # "compute_forces_numba_tiled": compute_forces_numba_tiled(args.threads),
                "compute_forces_numba_optimized": compute_forces_numba_optimized(args.threads),
            }
        },
        "triton": {
            "measure": measure_time_triton,
            "kernels": {
                "compute_accel_triton_naive": compute_accel_triton_naive,
                "compute_accel_triton_optimized": compute_accel_triton_optimized,
                # "compute_accel_triton_tensor": compute_accel_triton_tensor,
                # "compute_accel_triton_tiled": compute_accel_triton_tiled,
                # "compute_accel_triton_mixed": compute_accel_triton_mixed,
            }
        },
        "pytorch": {
            "measure": measure_time_torch,
            "kernels": {
                "compute_forces_pytorch_naive":     compute_forces_pytorch_naive,
                "compute_forces_pytorch_keops":     compute_forces_pytorch_keops,
                # "compute_forces_pytorch_chunked":   compute_forces_pytorch_chunked,
                # "compute_forces_pytorch_matmul":    compute_forces_pytorch_matmul,
                # "compute_forces_pytorch_optimized": compute_forces_pytorch_optimized,
                }
        }
    }

    print("START SCALING BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")

    n_particles = [(4 * i)**3 for i in range(args.n_start, args.n_end + 1)]

    for force_func_str in args.force_func: 
        if "numba" in force_func_str:
            framework = "numba"
        elif "cupy" in force_func_str:
            framework = "cupy"
        elif "pytorch" in force_func_str:
            framework = "pytorch"
        elif "triton" in force_func_str:
            framework = "triton"

        config = FRAMEWORK_CONFIG[framework]
        print(f"Measure {force_func_str.capitalize()}...")

        force_func = config["kernels"][force_func_str]

        # Framework-specific arguments
        measure_kwargs = {}
        if framework == "triton":
            measure_kwargs["block_size"] = args.bs_triton
        elif framework == "cupy":
            measure_kwargs["threads"] = args.threads

        results = run_scaling_benchmark(
            config["measure"], 
            n_particles, 
            compute_forces=force_func, 
            dt=args.dt, 
            steps=args.steps, **measure_kwargs
        )

        if args.store_results or args.store_plot: report_folder, timestamp = create_report(force_func_str)
        if args.store_results: store_results(force_func_str, results, timestamp, report_folder)
        if args.store_plot: plot_results(force_func_str, results["num_bodies"], results["interactions_per_second"], report_folder)

        cleanup_gpu()
        if len(args.force_func) > 1 : print("-" * 20 + "\n")

    print("END SCALING BENCHMARK")
    print("-" * 40 + "\n" + "-" * 40 + "\n")