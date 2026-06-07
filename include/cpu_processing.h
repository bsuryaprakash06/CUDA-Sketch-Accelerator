#ifndef CPU_PROCESSING_H
#define CPU_PROCESSING_H

#include <opencv2/opencv.hpp>

class CpuImageProcessor {
public:
    static void rgbToGrayscale(const cv::Mat& input, cv::Mat& output);
    static void gaussianBlur(const cv::Mat& input, cv::Mat& output, int kernelSize = 5);
    static void edgeDetectionSobel(const cv::Mat& input, cv::Mat& output);
    static void sharpen(const cv::Mat& input, cv::Mat& output);
    static void imageNegative(const cv::Mat& input, cv::Mat& output);
    static void contrastEnhancement(const cv::Mat& input, cv::Mat& output, float alpha, int beta);
    static void createPencilSketch(const cv::Mat& input, cv::Mat& output);
};

#endif // CPU_PROCESSING_H
