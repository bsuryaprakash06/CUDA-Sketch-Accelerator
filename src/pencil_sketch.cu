#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 16

// Device helper for Color Dodge blend
__device__ unsigned char colorDodgeBlend(unsigned char bottom, unsigned char top) {
    if (top == 255) return 255;
    int val = (bottom << 8) / (255 - top);
    if (val > 255) return 255;
    return (unsigned char)val;
}

// Color Dodge Kernel
__global__ void colorDodgeBlendKernel(const unsigned char* d_gray, const unsigned char* d_blur, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x;
        d_output[index] = colorDodgeBlend(d_gray[index], d_blur[index]);
    }
}

// Pencil sketch pipeline execution
// This function orchestrates the multiple kernel launches required for the pencil sketch
void launchPencilSketch(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    // 1. Convert to grayscale
    unsigned char *d_gray;
    cudaMalloc(&d_gray, width * height * sizeof(unsigned char));
    launchRgbToGrayscale(d_input, d_gray, width, height, channels);
    
    // 2. Invert grayscale
    unsigned char *d_inverted;
    cudaMalloc(&d_inverted, width * height * sizeof(unsigned char));
    launchImageNegative(d_gray, d_inverted, width, height, 1);

    // 3. Gaussian Blur on inverted image
    unsigned char *d_blur;
    cudaMalloc(&d_blur, width * height * sizeof(unsigned char));
    launchGaussianBlur(d_inverted, d_blur, width, height);

    // 4. Color Dodge Blend
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    colorDodgeBlendKernel<<<grid, block>>>(d_gray, d_blur, d_output, width, height);

    // Free intermediates
    cudaFree(d_gray);
    cudaFree(d_inverted);
    cudaFree(d_blur);
}
