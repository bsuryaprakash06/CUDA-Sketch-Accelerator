# Usage Instructions

Once you have successfully built the project, you can run the executable to process an image and compare CPU vs GPU performance.

## Running the Executable

The program requires the path to an input image.

### Syntax
```bash
./CudaImageEnhancement <path_to_input_image>
```

### Example
From the `build` directory (or `build/Release` on Windows):
```bash
./CudaImageEnhancement ../images/sample1.jpg
```

### What Happens When You Run It?
1. The program loads the image into memory using OpenCV.
2. It allocates memory on the GPU, using **pinned memory** for optimized transfer speeds.
3. It runs a series of image processing filters (Grayscale, Blur, Sobel, Sharpen, Negative, Contrast) first on the **CPU**, then on the **GPU**.
4. It executes the **Pencil Sketch pipeline** on both CPU and GPU.
5. All processed images are saved to the `output/` directory in the root of the project.
6. Execution times are measured and displayed in the console.
7. A file named `performance_metrics.csv` is generated in the `results/` directory, detailing the execution times and calculated speedups.

## Generating Performance Graphs

If you have Python installed, you can generate visual graphs comparing CPU and GPU performance based on the CSV data.

1. Install dependencies:
   ```bash
   pip install matplotlib pandas
   ```

2. Run the script from the root directory:
   ```bash
   python scripts/plot_performance.py
   ```

This will read `results/performance_metrics.csv` and generate bar charts (saved as PNGs in the `results/` folder) showing the execution time differences and speedup factors for each operation.
