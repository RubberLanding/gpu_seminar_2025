import csv
import torch
import datetime 
import cupy as cp
import matplotlib.pyplot as plt

from pathlib import Path

def print_results(total_time, steps_per_second, interactions_per_second):
    print("-" * 30)
    print(f"Total Runtime:            {total_time:.4f} seconds")
    print(f"Performance Steps:        {steps_per_second:.2f} steps/second")
    print(f"Performance Interactions: {interactions_per_second:.2f} interactions/second")
    print("-" * 30, "\n")

def cleanup_gpu():
    """Clears memory for all libraries to ensure a fair start for the next benchmark."""
    # Clear PyTorch
    if 'torch' in globals():
        torch.cuda.empty_cache()
    
    # Clear CuPy
    if 'cp' in globals():
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    # General Python Garbage Collection
    import gc
    gc.collect()

def create_report(method, report_base_dir="scaling_reports"):
    """
    Creates a folder and saves the benchmark data as a CSV using Pathlib.
    """
    # Create the timestamped folder path object
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_folder = Path(report_base_dir) / f"{method}_{timestamp}"
    
    # mkdir -p: parents=True ensures 'reports/' is created if it doesn't exist
    report_folder.mkdir(parents=True, exist_ok=True)

    return report_folder

def store_results(report_folder, n_values, interactions_values):
    file_path = report_folder / "scaling_results.csv"
    with file_path.open(mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Interactions_Per_Sec"])
        for n, interaction in zip(n_values, interactions_values):
            writer.writerow([n, interaction])  
    print(f"\n[✔] Benchmark data saved to: {file_path.resolve()}")

    return n_values, interactions_values

def plot_results(args, n_values, interactions_values, report_folder):
    """
    Plots N vs Interactions Per Second.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Data
    plt.plot(n_values, interactions_values, marker='o', linestyle='-', color='b', label=f'{args.method}')
    
    # Formatting
    plt.title('N-Body Simulation Performance: Scaling with N', fontsize=14)
    plt.xlabel('Number of Bodies (N)', fontsize=12)
    plt.ylabel('Interactions per Second', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Use Scientific Notation for Y-axis if numbers are huge
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Save and Show
    img_path = report_folder / f'nbody_scaling_{args.method}.png'
    plt.savefig(img_path)
    print(f"\nPlot saved to {img_path}")
    # plt.show() # Uncomment if running locally with a display

