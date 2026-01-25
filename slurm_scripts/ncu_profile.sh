#!/bin/bash
#SBATCH --job-name=nbody_profile_ncu
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --account=training2558
#SBATCH --output=/p/home/jusers/%u/jureca/gpu_seminar_2025/slurm_reports/ncu/profile_output_%j.txt
#SBATCH --error=/p/home/jusers/%u/jureca/gpu_seminar_2025/slurm_reports/ncu/profile_error_%j.txt

# Set MODE to the first argument, or default to "numba" if not provided
MODE=${1:-numba}
echo "Submitting job for mode: $MODE"

REPORT_DIR="$HOME/gpu_seminar_2025/profiling_reports/ncu"
mkdir -p "$REPORT_DIR" # Ensure the directory exists
echo "Reports will be saved to: $REPORT_DIR"

SCRIPT_PATH="$HOME/gpu_seminar_2025/src/nbody/${MODE}_/simulation.py"
# Check if file exists to prevent confusing errors later
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: File $SCRIPT_PATH does not exist!"
    exit 1
fi
echo "Running script: $SCRIPT_PATH"

# Load necessary modules
module purge
module load Stages/2025 GCCcore/.13.3.0
module load Nsight-Compute/2024.3.2
module load CUDA # Necessary for PyKeOps when simulating with Pytorch

# Activate environment
source ~/.bashrc  
micromamba activate nbody

# Debug Info 
echo "Job running on node: $SLURMD_NODENAME"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

echo "Starting Nsight Compute for $MODE..."

ncu --set default \
    --section SpeedOfLight \
    --launch-skip 5 \
    --launch-count 1 \
    --target-processes all \
    -o "${REPORT_DIR}/nbody_profile_ncu_${MODE}_${SLURM_JOB_ID}" \
    --force-overwrite \
    python "$SCRIPT_PATH" -n 20000 --steps 5

echo "Profiling finished."