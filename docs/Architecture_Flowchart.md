# Architecture and Flowcharts

## 1. High-Level Architecture

The system is split between the Host (CPU) and the Device (GPU). The Host manages file I/O, memory allocation, and orchestrates the kernel launches.

```mermaid
graph TD
    A[Input Image] -->|OpenCV imread| B(Host Memory - Pinned)
    B -->|cudaMemcpy H2D| C(Device Global Memory)
    
    subgraph GPU Device
    C --> D[Grayscale Kernel]
    C --> E[Blur Kernel]
    C --> F[Sobel Kernel]
    C --> G[Pencil Sketch Pipeline]
    end
    
    D -->|cudaMemcpy D2H| H(Host Memory)
    E -->|cudaMemcpy D2H| H
    F -->|cudaMemcpy D2H| H
    G -->|cudaMemcpy D2H| H
    
    H -->|OpenCV imwrite| I[Output Images]
    
    subgraph Profiling
    J[CUDA Events] -.->|Measure Time| K[CSV Logger]
    K --> L[Python Matplotlib]
    L --> M[Performance Graphs]
    end
```

## 2. Pencil Sketch Pipeline Flowchart

The Pencil Sketch effect requires four distinct steps, chained together on the GPU.

```mermaid
graph TD
    Start((Start)) --> Load[Load RGB Image to Device]
    Load --> Gray[Kernel: RGB to Grayscale]
    
    Gray --> Invert[Kernel: Image Negative]
    Invert --> Blur[Kernel: Gaussian Blur]
    
    Gray --> Blend[Kernel: Color Dodge Blend]
    Blur --> Blend
    
    Blend --> Store[Copy Result to Host]
    Store --> End((End))
```

## 3. Thread Mapping Strategy

For all 2D image processing kernels, we map the execution grid directly to the image coordinates.

*   `x = blockIdx.x * blockDim.x + threadIdx.x`
*   `y = blockIdx.y * blockDim.y + threadIdx.y`
*   Block Size: 16x16 threads
*   Grid Size: `Ceil(Width/16)` x `Ceil(Height/16)` blocks.
