// utils.cpp
#include "utils.h"
#include <iostream>
#include <iomanip>

void Timer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    stop_time_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsedMilliseconds() const {
    return std::chrono::duration<double, std::milli>(stop_time_ - start_time_).count();
}

double Timer::elapsedSeconds() const {
    return std::chrono::duration<double>(stop_time_ - start_time_).count();
}

std::string formatBytes(size_t bytes) {
    static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffixIndex = 0;
    double count = static_cast<double>(bytes);

    while (count >= 1024 && suffixIndex < 4) {
        count /= 1024;
        ++suffixIndex;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << count << " " << suffixes[suffixIndex];
    return oss.str();
}