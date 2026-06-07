cmake_minimum_required(VERSION 3.18)
project(CudaImageEnhancement VERSION 1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# CUDA configurations
set(CMAKE_CUDA_ARCHITECTURES "all") # Or specify e.g., 75;80;86 for specific architectures
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/utils.cpp
    src/cpu_processing.cpp
)

set(CUDA_SOURCES
    src/gpu_kernels.cu
    src/pencil_sketch.cu
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Create output directories if they don't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)
