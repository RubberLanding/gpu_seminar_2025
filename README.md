# ✨ GPU N-Body Problem Solver

This repository was created for a seminar on efficient programming for GPUs. It contains efficient Python code to solve the **N-body problem** by leveraging **GPU acceleration** through various frameworks. Currently, the implementation focuses on using **Numba**. The code has been tested on CPU environments for initial verification.

-----

## 🚀 Frameworks

The goal of this project is to provide flexible implementations across different popular GPU frameworks. Below is the structure for the currently implemented and planned solutions.

### Numba 🐍 (Implemented)

This implementation uses **Numba** (specifically `numba.cuda`) to compile Python code for execution directly on NVIDIA GPUs.

  * **Status:** **Implemented**
  * **Location:** `nbody_numba.py` (or similar file)
  * **Key Features:** Utilizes JIT compilation for high-performance CUDA kernels.

-----

### CuPy 🌌 (Template)

*(This section is a template for future implementation)*

This implementation will use the **CuPy** library, which provides a NumPy-compatible array interface for GPU computation.

  * **Status:** Planned
-----

### PyTorch 🔥 (Template)

*(This section is a template for future implementation)*

This implementation will use **PyTorch** tensors and its CUDA backend. While primarily known for deep learning, PyTorch offers robust and efficient general-purpose GPU computing.

  * **Status:** Planned
-----


## 📦 Installation

Follow these steps to get the simulation running on your machine.

### 1. Prerequisites
Before you start, ensure you have the following installed:
* **Anaconda or Miniconda:** This manages the Python environment. [Download here](https://docs.conda.io/en/latest/miniconda.html).
* **Git:** To download the code. [Download here](https://git-scm.com/downloads).

### 2. Get the Code
Open your terminal (Mac/Linux) or Anaconda Prompt (Windows) and run the following command to download the project to your computer:
```bash
git clone [https://github.com/RubberLanding/gpu_seminar_2025.git](https://github.com/RubberLanding/gpu_seminar_2025.git)
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
python src/nbody/numba/simulation.py
```
