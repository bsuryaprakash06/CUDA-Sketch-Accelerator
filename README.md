# GPU-Accelerated Image Enhancement and Artistic Sketch Generation

This is a comprehensive C++/CUDA capstone project demonstrating how GPU parallelism accelerates common image processing tasks compared to a standard CPU implementation. The project includes several image enhancement filters and a multi-stage pipeline to generate a "Pencil Sketch" effect.

## Overview

The application processes input images using both CPU-based (OpenCV) and GPU-based (Custom CUDA Kernels) algorithms to demonstrate the significant speedup achieved by leveraging NVIDIA GPUs. 

### Features Implemented:
*   **RGB to Grayscale Conversion**
*   **Gaussian Blur**
*   **Edge Detection (Sobel Operator)**
*   **Sharpening Filter**
*   **Image Negative / Inversion**
*   **Contrast Enhancement**
*   **Pencil Sketch Effect** (A pipeline combining Grayscale, Blur, Inversion, and Dodge Blending)

### Key CUDA Concepts Demonstrated:
*   Custom kernels for pixel-wise and neighborhood operations
*   Thread hierarchy (Grids and Blocks) mapping to 2D image coordinates
*   Host-to-Device and Device-to-Host memory transfers
*   CUDA Events for accurate profiling and timing
*   Pinned (Page-locked) memory to accelerate data transfers

## Project Structure

```text
cuda-image-enhancement-capstone/
├── CMakeLists.txt         # Build configuration
├── README.md              # Project overview
├── INSTALL.md             # Installation instructions
├── USAGE.md               # How to run the application
├── include/               # C++/CUDA header files
├── src/                   # Source code (C++ and CUDA)
├── scripts/               # Python scripts for data visualization
├── images/                # Sample input images
├── output/                # Generated enhanced/sketch images (git-ignored)
├── results/               # CSV files and performance graphs (git-ignored)
└── docs/                  # Architecture diagrams, flowcharts, and reports
```

## Quick Start

Please refer to [INSTALL.md](INSTALL.md) for detailed prerequisites and build instructions.
For instructions on how to use the executable and generate graphs, see [USAGE.md](USAGE.md).

## Performance

The application automatically logs the execution times for both CPU and GPU implementations across all filters. The results are saved to a CSV file in the `results/` directory, which can be visualized using the provided Python script.
