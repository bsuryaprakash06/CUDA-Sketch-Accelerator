#include "utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>

void PerformanceLogger::addRecord(const std::string& name, int w, int h, float cpuTime, float gpuTime) {
    float speedup = (gpuTime > 0.0f) ? (cpuTime / gpuTime) : 0.0f;
    records.push_back({name, w, h, cpuTime, gpuTime, speedup});
}

void PerformanceLogger::saveToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }

    file << "Operation,Image_Width,Image_Height,CPU_Time_ms,GPU_Time_ms,Speedup\n";
    for (const auto& rec : records) {
        file << rec.operationName << ","
             << rec.imageWidth << ","
             << rec.imageHeight << ","
             << std::fixed << std::setprecision(3) << rec.cpuTimeMs << ","
             << rec.gpuTimeMs << ","
             << rec.speedup << "\n";
    }
    file.close();
    std::cout << "Performance metrics saved to " << filename << std::endl;
}

void PerformanceLogger::printSummary() const {
    std::cout << "\n================ Performance Summary ================\n";
    std::cout << std::left << std::setw(20) << "Operation" 
              << std::setw(15) << "CPU Time (ms)" 
              << std::setw(15) << "GPU Time (ms)" 
              << std::setw(10) << "Speedup" << "\n";
    std::cout << "-----------------------------------------------------\n";
    for (const auto& rec : records) {
        std::cout << std::left << std::setw(20) << rec.operationName
                  << std::setw(15) << std::fixed << std::setprecision(3) << rec.cpuTimeMs
                  << std::setw(15) << rec.gpuTimeMs
                  << std::setw(10) << rec.speedup << "x\n";
    }
    std::cout << "=====================================================\n";
}
