# CUDA Vector Addition with PyTorch

A demonstration of custom CUDA kernel development using PyTorch's C++/CUDA extension API for vector addition operations.

## Overview

This project implements a simple but complete example of writing, compiling, and benchmarking a custom CUDA kernel that performs element-wise vector addition. It showcases how to integrate low-level CUDA code with PyTorch tensors.

## Files

- **`addition.cu`** - CUDA kernel implementation with PyTorch C++ extensions
  - Custom CUDA kernel for parallel vector addition
  - Input validation macros
  - PyBind11 bindings for Python integration

- **`addition2.cu`** - Alternative/extended CUDA implementation

- **`main.py`** - Python wrapper script for loading and testing the CUDA extension
  - JIT compilation using `torch.utils.cpp_extension.load()`
  - Basic functionality test with random vectors

- **`cuda_vector_addition.ipynb`** - Complete Jupyter notebook for Google Colab
  - Self-contained implementation (no external files needed)
  - Step-by-step tutorial from installation to benchmarking
  - Works directly in Google Colab with GPU runtime


## Requirements

- Python 3.7+
- PyTorch with CUDA support
- NVIDIA GPU with CUDA toolkit
- C++ compiler (MSVC on Windows, GCC/Clang on Linux)

## Usage

### Local Development

1. **Configure VS Code** (Windows):

2. **Run the example**:
   ```bash
   python main.py
   ```

### Google Colab

1. Upload `cuda_vector_addition.ipynb` to Google Colab
2. Select GPU runtime: `Runtime → Change runtime type → GPU`
3. Run all cells sequentially

The notebook will:
- Check CUDA availability
- Write and compile the CUDA kernel
- Test vector addition on GPU
- Verify correctness against PyTorch's built-in operations
- Benchmark performance using CUDA events

## Implementation Details

### CUDA Kernel

The kernel uses a standard parallel reduction pattern:
- **Block size**: 256 threads per block
- **Grid size**: Dynamically calculated based on input size
- **Memory access**: Coalesced reads and writes for optimal performance

### Key Features

- Input validation (CUDA device check, contiguous memory)
- Automatic device synchronization
- Proper error handling with PyTorch's `TORCH_CHECK`
- Performance profiling with CUDA events

## Performance

The custom kernel should have comparable performance to PyTorch's built-in addition operator for large vectors, demonstrating that well-written custom CUDA code can match highly optimized library implementations.

## Learning Objectives

This project demonstrates:
- Writing CUDA kernels with `__global__` functions
- Integrating CUDA with PyTorch tensors
- Using PyTorch's C++ extension API
- JIT compilation of CUDA code
- Proper CUDA profiling techniques
- Memory management between CPU and GPU

## Notes

- First compilation may take several minutes as PyTorch compiles the CUDA code
- Subsequent runs use cached compilation
- For best performance, ensure tensors are contiguous in memory
- The kernel assumes `float` tensors; modification needed for other dtypes

## License

This is an educational example for learning CUDA programming with PyTorch.
