#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

// Struct to hold performance data for a specific operation
struct PerformanceRecord {
    std::string operationName;
    int imageWidth;
    int imageHeight;
    float cpuTimeMs;
    float gpuTimeMs;
    float speedup;
};

// Class to manage logging and CSV generation
class PerformanceLogger {
private:
    std::vector<PerformanceRecord> records;

public:
    void addRecord(const std::string& name, int w, int h, float cpuTime, float gpuTime);
    void saveToCSV(const std::string& filename) const;
    void printSummary() const;
};

#endif // UTILS_H
