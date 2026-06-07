#ifndef GPU_KERNELS_CUH
#define GPU_KERNELS_CUH

// CUDA kernels and wrapper functions for image processing
void launchRgbToGrayscale(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
void launchGaussianBlur(const unsigned char* d_input, unsigned char* d_output, int width, int height);
void launchSobelEdgeDetection(const unsigned char* d_input, unsigned char* d_output, int width, int height);
void launchSharpen(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
void launchImageNegative(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);
void launchContrastEnhancement(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels, float alpha, int beta);
void launchPencilSketch(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);

#endif // GPU_KERNELS_CUH
