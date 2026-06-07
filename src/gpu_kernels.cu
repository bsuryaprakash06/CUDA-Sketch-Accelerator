#include "gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 16

// Helper clamp function for device
__device__ unsigned char clamp_pixel(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return (unsigned char)value;
}

// ---------------------------------------------------------
// RGB to Grayscale
// ---------------------------------------------------------
__global__ void rgbToGrayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = (y * width + x) * channels;
        int outIndex = y * width + x;
        // OpenCV uses BGR format
        unsigned char b = input[index];
        unsigned char g = input[index + 1];
        unsigned char r = input[index + 2];
        
        output[outIndex] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void launchRgbToGrayscale(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbToGrayscaleKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
}

// ---------------------------------------------------------
// Gaussian Blur (simplified 5x5)
// ---------------------------------------------------------
__constant__ float d_gaussianKernel[25] = {
    1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f,
    4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
    7/273.0f, 26/273.0f, 41/273.0f, 26/273.0f, 7/273.0f,
    4/273.0f, 16/273.0f, 26/273.0f, 16/273.0f, 4/273.0f,
    1/273.0f, 4/273.0f, 7/273.0f, 4/273.0f, 1/273.0f
};

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = 2;

    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        float sum = 0.0f;
        int kIndex = 0;
        for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
                sum += input[(y + ky) * width + (x + kx)] * d_gaussianKernel[kIndex++];
            }
        }
        output[y * width + x] = clamp_pixel((int)sum);
    } else if (x < width && y < height) {
        // Border pixels remain unchanged
        output[y * width + x] = input[y * width + x];
    }
}

void launchGaussianBlur(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    gaussianBlurKernel<<<grid, block>>>(d_input, d_output, width, height);
}

// ---------------------------------------------------------
// Sobel Edge Detection
// ---------------------------------------------------------
__global__ void sobelEdgeDetectionKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
                 -2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)]
                 -input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

        int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
                 +input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

        int sum = abs(gx) / 2 + abs(gy) / 2;
        output[y * width + x] = clamp_pixel(sum);
    } else if (x < width && y < height) {
        output[y * width + x] = 0;
    }
}

void launchSobelEdgeDetection(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    sobelEdgeDetectionKernel<<<grid, block>>>(d_input, d_output, width, height);
}

// ---------------------------------------------------------
// Sharpen
// ---------------------------------------------------------
__global__ void sharpenKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        for (int c = 0; c < channels; c++) {
            int center = input[(y * width + x) * channels + c];
            int top = input[((y - 1) * width + x) * channels + c];
            int bottom = input[((y + 1) * width + x) * channels + c];
            int left = input[(y * width + (x - 1)) * channels + c];
            int right = input[(y * width + (x + 1)) * channels + c];

            int sum = 5 * center - top - bottom - left - right;
            output[(y * width + x) * channels + c] = clamp_pixel(sum);
        }
    } else if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            output[(y * width + x) * channels + c] = input[(y * width + x) * channels + c];
        }
    }
}

void launchSharpen(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    sharpenKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
}

// ---------------------------------------------------------
// Image Negative
// ---------------------------------------------------------
__global__ void imageNegativeKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            int index = (y * width + x) * channels + c;
            output[index] = 255 - input[index];
        }
    }
}

void launchImageNegative(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    imageNegativeKernel<<<grid, block>>>(d_input, d_output, width, height, channels);
}

// ---------------------------------------------------------
// Contrast Enhancement
// ---------------------------------------------------------
__global__ void contrastEnhancementKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels, float alpha, int beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            int index = (y * width + x) * channels + c;
            float val = alpha * input[index] + beta;
            output[index] = clamp_pixel((int)val);
        }
    }
}

void launchContrastEnhancement(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float alpha, int beta) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    contrastEnhancementKernel<<<grid, block>>>(d_input, d_output, width, height, channels, alpha, beta);
}
