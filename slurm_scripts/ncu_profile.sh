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

MODE=${1:-numba}
NUM_PARTICLES=${2:-100000}
NUM_STEPS=${3:-10} 
FORCE_FUNC=${4:-compute_forces_numba_tiled} 

# Set a base directory to ensure relative execution paths work correctly
BASE_DIR="$HOME/gpu_seminar_2025"
REPORT_DIR="${BASE_DIR}/profiling_reports/ncu"
SCRIPT_PATH="${BASE_DIR}/src/nbody/${MODE}_/simulation.py"

mkdir -p "$REPORT_DIR"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: File $SCRIPT_PATH does not exist!"
    exit 1
fi

cd "$BASE_DIR" || exit 1

echo "Submitting job for mode:   $MODE"
echo "Job running on node:       $SLURMD_NODENAME"
echo "GPUs available:            $CUDA_VISIBLE_DEVICES"

# --- Module Loading ---
module purge
module load Stages/2025 
module load GCCcore/.13.3.0
module load Nsight-Compute/2024.3.2
module load CUDA

source ~/.bashrc  
micromamba activate nbody

NCU_EXTRA_FLAGS=""

case $MODE in
    "numba")
        NCU_EXTRA_FLAGS="-k regex:compute_forces.* --import-source yes --resolve-source-file \"$SCRIPT_PATH\""
        echo "Profiling Numba with LineInfo support..."
        ;;  

    "cupy")
        export CUPY_CACHE_SAVE_CUDA_SOURCE=1
        NCU_EXTRA_FLAGS="-k regex:compute_forces.* --import-source yes --source-folder ~/.cupy/kernel_cache" 
        echo "Profiling CuPy with C++ source caching and Debug info..."
        ;;     

    "pytorch")

        # --nvtx: Tells NCU to respect the markers in your code.
        # --nvtx-include: Limits profiling to the actual simulation step.
        # Note: Ensure your PyTorch code uses torch.cuda.nvtx.range_push("nbody_step")
        NCU_EXTRA_FLAGS="--nvtx --nvtx-include nbody_step/"
        echo "Profiling PyTorch with NVTX range filtering..."
        ;;

    "triton")
        NCU_EXTRA_FLAGS="-k regex:compute_accel.* --import-source yes"
        echo "Profiling Triton..."
        ;;
esac

echo "Starting Nsight Compute for $MODE..."

# FORCE CUPY TO RECOMPILE: Clear the old cached kernels
if [ "$MODE" == "cupy" ]; then
    echo "Clearing CuPy kernel cache..."
    unset CUPY_CUDA_COMPILE_WITH_DEBUG
    rm -rf ~/.cupy/kernel_cache/*
    rm -rf ~/.nv/ComputeCache/*
fi

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
    -o "${REPORT_DIR}/nbody_profile_ncu_${MODE}_${SLURM_JOB_ID}" \
    --force-overwrite \
    python src/nbody/benchmark/benchmark.py \
        -f "$FORCE_FUNC" \
        -n "$NUM_PARTICLES" \
        -s "$NUM_STEPS"

echo "Profiling finished. Report saved to: $REPORT_DIR"