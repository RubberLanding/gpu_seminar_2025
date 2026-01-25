#!/bin/bash
#SBATCH --job-name=nbody_profile_nsys
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --account=training2558
#SBATCH --output=/p/home/jusers/%u/jureca/gpu_seminar_2025/slurm_reports/nsys/profile_output_%j.txt
#SBATCH --error=/p/home/jusers/%u/jureca/gpu_seminar_2025/slurm_reports/nsys/profile_error_%j.txt

# Set MODE to the first argument, or default to "numba" if not provided
MODE=${1:-numba}
echo "Submitting job for mode: $MODE"

REPORT_DIR="$HOME/gpu_seminar_2025/profiling_reports/nsys"
mkdir -p "$REPORT_DIR" # Ensure the directory exists
echo "Reports will be saved to: $REPORT_DIR"

SCRIPT_PATH="src/nbody/${MODE}_/simulation.py"
# Check if file exists to prevent confusing errors later
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: File $SCRIPT_PATH does not exist!"
    exit 1
fi
echo "Running script: $SCRIPT_PATH"

# Load necessary modules
module purge
module load Stages/2025 GCCcore/.13.3.0
module load Nsight-Systems/2024.7.1
module load CUDA # Necessary for PyKeOps when simulating with Pytorch

# Activate environment
source ~/.bashrc  
micromamba activate nbody

# Debug Info 
echo "Job running on node: $SLURMD_NODENAME"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

echo "Starting Nsight Systems for $MODE..."

nsys profile \
  --trace=cuda,nvtx \
  --stats=true \
  -o "${REPORT_DIR}/nbody_profile_nsys_${MODE}_${SLURM_JOB_ID}" \
  --force-overwrite=true \
  python "$SCRIPT_PATH" -n 50000 -s 30

echo "Profiling finished."