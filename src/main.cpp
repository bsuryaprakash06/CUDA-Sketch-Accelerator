#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include "cpu_processing.h"
#include "gpu_kernels.cuh"
#include "utils.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image from " << imagePath << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    size_t imgSize = width * height * channels;

    std::cout << "Loaded image: " << imagePath << " (" << width << "x" << height << ")" << std::endl;

    PerformanceLogger logger;

    // Output mats
    cv::Mat cpuGray, cpuBlur, cpuSobel, cpuSharpen, cpuNegative, cpuContrast, cpuSketch;
    cv::Mat gpuGray(height, width, CV_8UC1);
    cv::Mat gpuBlur(height, width, CV_8UC1);
    cv::Mat gpuSobel(height, width, CV_8UC1);
    cv::Mat gpuSharpen(height, width, CV_8UC3);
    cv::Mat gpuNegative(height, width, CV_8UC3);
    cv::Mat gpuContrast(height, width, CV_8UC3);
    cv::Mat gpuSketch(height, width, CV_8UC1);

    // -------------------------------------------------------------
    // CUDA Pinned Memory Allocation
    // -------------------------------------------------------------
    unsigned char* h_inputPinned;
    cudaMallocHost((void**)&h_inputPinned, imgSize);
    std::memcpy(h_inputPinned, inputImage.data, imgSize);

    // Device allocations
    unsigned char *d_input, *d_output1, *d_output3;
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output1, width * height); // 1 channel
    cudaMalloc(&d_output3, imgSize);        // 3 channels

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpuTimeMs = 0.0f;

    // Data Transfer (Host to Device)
    cudaEventRecord(start);
    cudaMemcpy(d_input, h_inputPinned, imgSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    logger.addRecord("H2D_Transfer", width, height, 0.0f, gpuTimeMs);

    auto measureCpu = [](auto func, float& cpuTime) {
        auto t1 = std::chrono::high_resolution_clock::now();
        func();
        auto t2 = std::chrono::high_resolution_clock::now();
        cpuTime = std::chrono::duration<float, std::milli>(t2 - t1).count();
    };

    float cpuMs = 0;

    // --- 1. Grayscale ---
    measureCpu([&]() { CpuImageProcessor::rgbToGrayscale(inputImage, cpuGray); }, cpuMs);
    cudaEventRecord(start);
    launchRgbToGrayscale(d_input, d_output1, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    cudaMemcpy(gpuGray.data, d_output1, width * height, cudaMemcpyDeviceToHost);
    logger.addRecord("Grayscale", width, height, cpuMs, gpuTimeMs);

    // --- 2. Gaussian Blur ---
    measureCpu([&]() { CpuImageProcessor::gaussianBlur(cpuGray, cpuBlur); }, cpuMs); // run on gray for simplicity
    cudaEventRecord(start);
    launchGaussianBlur(d_output1, d_output1, width, height); // in-place or separate, using d_output1 as input for demo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    cudaMemcpy(gpuBlur.data, d_output1, width * height, cudaMemcpyDeviceToHost);
    logger.addRecord("Gaussian_Blur", width, height, cpuMs, gpuTimeMs);

    // --- 3. Sobel Edge Detection ---
    measureCpu([&]() { CpuImageProcessor::edgeDetectionSobel(cpuGray, cpuSobel); }, cpuMs);
    // Reload gray into d_output1 for sobel input
    cudaMemcpy(d_output1, gpuGray.data, width*height, cudaMemcpyHostToDevice);
    unsigned char* d_sobelOut; cudaMalloc(&d_sobelOut, width * height);
    cudaEventRecord(start);
    launchSobelEdgeDetection(d_output1, d_sobelOut, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    cudaMemcpy(gpuSobel.data, d_sobelOut, width * height, cudaMemcpyDeviceToHost);
    cudaFree(d_sobelOut);
    logger.addRecord("Sobel_Edge", width, height, cpuMs, gpuTimeMs);

    // --- 4. Sharpen ---
    measureCpu([&]() { CpuImageProcessor::sharpen(inputImage, cpuSharpen); }, cpuMs);
    cudaEventRecord(start);
    launchSharpen(d_input, d_output3, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    cudaMemcpy(gpuSharpen.data, d_output3, imgSize, cudaMemcpyDeviceToHost);
    logger.addRecord("Sharpen", width, height, cpuMs, gpuTimeMs);

    // --- 5. Image Negative ---
    measureCpu([&]() { CpuImageProcessor::imageNegative(inputImage, cpuNegative); }, cpuMs);
    cudaEventRecord(start);
    launchImageNegative(d_input, d_output3, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    cudaMemcpy(gpuNegative.data, d_output3, imgSize, cudaMemcpyDeviceToHost);
    logger.addRecord("Image_Negative", width, height, cpuMs, gpuTimeMs);

    // --- 6. Contrast Enhancement ---
    measureCpu([&]() { CpuImageProcessor::contrastEnhancement(inputImage, cpuContrast, 1.5f, 20); }, cpuMs);
    cudaEventRecord(start);
    launchContrastEnhancement(d_input, d_output3, width, height, channels, 1.5f, 20);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    cudaMemcpy(gpuContrast.data, d_output3, imgSize, cudaMemcpyDeviceToHost);
    logger.addRecord("Contrast_Enhance", width, height, cpuMs, gpuTimeMs);

    // --- 7. Pencil Sketch Pipeline ---
    measureCpu([&]() { CpuImageProcessor::createPencilSketch(inputImage, cpuSketch); }, cpuMs);
    cudaEventRecord(start);
    launchPencilSketch(d_input, d_output1, width, height, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    cudaMemcpy(gpuSketch.data, d_output1, width * height, cudaMemcpyDeviceToHost);
    logger.addRecord("Pencil_Sketch", width, height, cpuMs, gpuTimeMs);

    // Save Output Images
    cv::imwrite("output/cpu_sketch.jpg", cpuSketch);
    cv::imwrite("output/gpu_sketch.jpg", gpuSketch);
    cv::imwrite("output/gpu_sharpen.jpg", gpuSharpen);
    cv::imwrite("output/gpu_edges.jpg", gpuSobel);
    std::cout << "Images saved to output/ directory." << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output3);
    cudaFreeHost(h_inputPinned);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Log & Save
    logger.printSummary();
    logger.saveToCSV("results/performance_metrics.csv");

    return 0;
}
