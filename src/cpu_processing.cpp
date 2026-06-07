#include "cpu_processing.h"

void CpuImageProcessor::rgbToGrayscale(const cv::Mat& input, cv::Mat& output) {
    cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
}

void CpuImageProcessor::gaussianBlur(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    cv::GaussianBlur(input, output, cv::Size(kernelSize, kernelSize), 0);
}

void CpuImageProcessor::edgeDetectionSobel(const cv::Mat& input, cv::Mat& output) {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    cv::Sobel(input, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(input, grad_y, CV_16S, 0, 1, 3);

    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, output);
}

void CpuImageProcessor::sharpen(const cv::Mat& input, cv::Mat& output) {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0);
    cv::filter2D(input, output, input.depth(), kernel);
}

void CpuImageProcessor::imageNegative(const cv::Mat& input, cv::Mat& output) {
    cv::bitwise_not(input, output);
}

void CpuImageProcessor::contrastEnhancement(const cv::Mat& input, cv::Mat& output, float alpha, int beta) {
    input.convertTo(output, -1, alpha, beta);
}

void CpuImageProcessor::createPencilSketch(const cv::Mat& input, cv::Mat& output) {
    cv::Mat gray, blur, inverted;
    // 1. Grayscale
    rgbToGrayscale(input, gray);
    // 2. Invert
    imageNegative(gray, inverted);
    // 3. Gaussian Blur
    gaussianBlur(inverted, blur, 21);
    // 4. Color Dodge Blend
    cv::divide(gray, 255 - blur, output, 256.0);
}
