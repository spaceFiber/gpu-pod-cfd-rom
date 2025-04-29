// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <chrono>
#include <string>

#define CUDA_CHECK(call) {                                          \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));         \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

class Timer {
public:
    void start();
    void stop();
    double elapsedMilliseconds() const;
    double elapsedSeconds() const;
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stop_time_;
};

std::string formatBytes(size_t bytes);

#endif // UTILS_H