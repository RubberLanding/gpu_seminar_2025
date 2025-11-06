# ✨ GPU N-Body Problem Solver

This repository contains efficient Python code to solve the **N-body problem** by leveraging **GPU acceleration** through various frameworks. Currently, the implementation focuses on using **Numba**. The code has been tested on CPU environments for initial verification.

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

