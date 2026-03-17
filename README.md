# ✨ GPU N-Body Problem Solver

This repository was created for a seminar on efficient programming for GPUs. It contains efficient Python code to solve the **N-body problem** by leveraging **GPU acceleration** through various frameworks. This project provides a comparative analysis of four prominent Python-based GPU programming frameworks: **CuPy, Numba, PyTorch, and Triton**. The code has been tested on modern GPU architectures to analyze the shifting bottlenecks between memory bandwidth and arithmetic throughput.

-----

## 🚀 Frameworks

The goal of this project is to provide flexible, high-performance implementations across different popular GPU frameworks to avoid vendor lock-in and evaluate hardware utilization. Below is the structure for the implemented solutions.

### Numba 🐍 (Implemented)

This implementation uses **Numba** (specifically `numba.cuda`) to compile pure Python code directly into optimized machine code or CUDA PTX.
* **Status:** Implemented
* **Key Features:** Utilizes JIT compilation for high-performance CUDA kernels, granting explicit control over thread hierarchies and shared memory.

-----

### CuPy 🌌 (Implemented)

This implementation uses the **CuPy** library, serving as a GPU-accelerated drop-in replacement for NumPy12, 34]. 
* **Status:** Implemented
* **Key Features:** Uses a hybrid approach, combining high-level array operations for integration steps with the injection of custom C++ CUDA strings (via `RawKernel`) for the $\mathcal{O}(N^2)$ pairwise force calculations.

-----

### PyTorch 🔥 (Implemented)

This implementation uses **PyTorch** tensors and its CUDA backend. While primarily known for deep learning, PyTorch offers robust and efficient general-purpose GPU computing.
* **Status:** Implemented
* **Key Features:** To overcome massive $\mathcal{O}(N^2)$ memory bandwidth bottlenecks and Out-Of-Memory errors during eager execution, this implementation utilizes **PyKeOps**. It leverages lazy evaluation and symbolic tensors to dynamically compile highly optimized C++ kernels on the fly.

-----

### Triton 🔱 (Implemented)

This implementation uses **Triton**, a specialized language and compiler designed to simplify the development of highly efficient GPU kernels.
* **Status:** Implemented
* **Key Features:** Introduces a unique block-based programming paradigm that abstracts away individual thread management. The Triton compiler automatically handles complex low-level tasks, including memory coalescing, synchronization, and shared memory utilization.

-----

## 📦 Installation

Follow these steps to get the simulation running on your machine. The Python environment and its dependencies are managed using `micromamba` (or `conda`)245].

### 1. Prerequisites
Before you start, ensure you have the following installed:
* **Anaconda / Miniconda / Micromamba:** This manages the Python environment. [Download Miniconda here](https://docs.conda.io/en/latest/miniconda.html).
* **Git:** To download the code. [Download here](https://git-scm.com/downloads).

### 2. Get the Code
Open your terminal (Mac/Linux) or Anaconda Prompt (Windows) and run the following commands to download the project to your computer246]:
```bash
git clone https://github.com/RubberLanding/gpu_seminar_2025.git
cd gpu_seminar_2025
```

Run this command to create the virtual environment (this might take a few minutes) and activate it:
```bash
conda env create -f environment.yaml
conda activate nbody_numba
```

Finally, install the project:
```bash
pip install .
```

To make sure that everything is running correctly, try running the simulation with a small number of particles, e.g. with Numba:
```bash
python src/nbody/numba_/simulation.py
```
