# Installation Guide

## Prerequisites

To build and run this project, your system must meet the following requirements:

1. **NVIDIA GPU**: You must have a CUDA-capable NVIDIA GPU.
2. **CUDA Toolkit**: Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
3. **C++ Compiler**: 
   - **Windows**: Install [Visual Studio](https://visualstudio.microsoft.com/) with "Desktop development with C++" workload.
   - **Linux**: Install `gcc` and `g++`.
4. **CMake**: Install [CMake](https://cmake.org/download/) (version 3.18 or higher).
5. **OpenCV**: 
   - Download [OpenCV](https://opencv.org/releases/) and extract it (e.g., to `C:\opencv`).
   - Set the `OpenCV_DIR` environment variable to the path containing `OpenCVConfig.cmake` (e.g., `C:\opencv\build`).
6. **Python (Optional)**: Required only if you want to generate the performance graphs from the output CSV.
   - Install `matplotlib` and `pandas` (`pip install matplotlib pandas`).

## Building the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/CUDA-Sketch-Accelerator.git
   cd CUDA-Sketch-Accelerator
   ```

2. **Generate Build Files via CMake**:
   Create a build directory and run CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   ```
   *Note for Windows users: You might need to specify the generator and architecture, e.g., `cmake -G "Visual Studio 17 2022" -A x64 ..`*

3. **Compile the Project**:
   ```bash
   cmake --build . --config Release
   ```

4. **Verify Build**:
   After compilation, the executable `CudaImageEnhancement` will be located in the `Release` folder (Windows) or the root of your `build` directory (Linux).
