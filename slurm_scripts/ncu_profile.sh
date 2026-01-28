#!/bin/bash
#SBATCH --job-name=nbody_profile_ncu
#SBATCH --account=training2558
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/p/home/jusers/%u/jureca/gpu_seminar_2025/slurm_reports/ncu/profile_output_%j.txt
#SBATCH --error=/p/home/jusers/%u/jureca/gpu_seminar_2025/slurm_reports/ncu/profile_error_%j.txt

# --- Environment Setup ---
# Default to pytorch if no argument is provided
MODE=${1:-pytorch}
NUM_PARTICLES=${2:-100000}
NUM_STEPS=${3:-10} 

REPORT_DIR="$HOME/gpu_seminar_2025/profiling_reports/ncu"
SCRIPT_PATH="$HOME/gpu_seminar_2025/src/nbody/${MODE}_/simulation.py"

mkdir -p "$REPORT_DIR"

# --- Validation ---
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: File $SCRIPT_PATH does not exist!"
    exit 1
fi

echo "Submitting job for mode: $MODE"
echo "Job running on node:    $SLURMD_NODENAME"
echo "GPUs available:         $CUDA_VISIBLE_DEVICES"

# --- Module Loading ---
module purge
module load Stages/2025 
module load GCCcore/.13.3.0
module load Nsight-Compute/2024.3.2
module load CUDA

source ~/.bashrc  
micromamba activate nbody

# --- Framework-Specific Configuration ---
NCU_EXTRA_FLAGS=""

case $MODE in
    "numba")
        # Use single quotes for the variable and escaped/nested quotes for the path
        NCU_EXTRA_FLAGS="-k regex:.*gpu_force_kernel.* --import-source yes --resolve-source-file \"$SCRIPT_PATH\""
        echo "Profiling Numba with LineInfo support..."
        ;;    
    "cupy")
    # Environment variable to prevent CuPy from deleting temp .cu files.
        export CUPY_CACHE_SAVE_CUDA_SOURCE=1
        NCU_EXTRA_FLAGS="--import-source yes"
        echo "Profiling CuPy with C++ source caching..."
        ;;
    "pytorch")
        # --nvtx: Tells NCU to respect the markers in your code.
        # --nvtx-include: Limits profiling to the actual simulation step.
        NCU_EXTRA_FLAGS="--nvtx --nvtx-include nbody_step/"
        echo "Profiling PyTorch with NVTX range filtering..."
        ;;
esac

# --- Execution ---
echo "Starting Nsight Compute for $MODE..."

ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section SchedulerStats \
    --section WarpStateStats \
    --section SourceCounters \
    --launch-skip 5 \
    --launch-count 1 \
    --target-processes all \
    $NCU_EXTRA_FLAGS \
    -o "${REPORT_DIR}/nbody_profile_nsys_${MODE}_${SLURM_JOB_ID}" \
    --force-overwrite \
    python "$SCRIPT_PATH" -n "$NUM_PARTICLES" --steps "$NUM_STEPS"

echo "Profiling finished. Report saved to: $REPORT_DIR"