// ml_enhancer.cpp
#include "ml_enhancer.h"
#include <iostream>
#include <random>
#include <algorithm>

MLEnhancer::MLEnhancer() {
    // Initialize random seed
    rng_ = std::mt19937(rd_());
    dist_ = std::uniform_real_distribution<float>(-ML_NOISE_LEVEL, ML_NOISE_LEVEL);
}

std::vector<std::vector<float>> MLEnhancer::enhanceModes(
    const std::vector<std::vector<float>>& podModes,
    const std::vector<std::vector<float>>& snapshots) 
{
    std::cout << "[MLEnhancer] Starting machine learning enhancement..." << std::endl;

    std::vector<std::vector<float>> enhancedModes = podModes;

    for (size_t modeIdx = 0; modeIdx < enhancedModes.size(); ++modeIdx) {
        std::vector<float>& mode = enhancedModes[modeIdx];

        // Example "learning" step: adjust mode based on correlation with snapshots
        for (size_t i = 0; i < mode.size(); ++i) {
            float adjustment = computeLearningAdjustment(i, mode, snapshots);
            mode[i] += adjustment;
        }

        // Normalize the enhanced mode
        float norm = computeNorm(mode);
        if (norm > 1e-8f) {
            for (float& val : mode) {
                val /= norm;
            }
        }
    }

    std::cout << "[MLEnhancer] Enhancement complete." << std::endl;
    return enhancedModes;
}

float MLEnhancer::computeLearningAdjustment(
    size_t idx,
    const std::vector<float>& mode,
    const std::vector<std::vector<float>>& snapshots) 
{
    float sum = 0.0f;
    for (const auto& snapshot : snapshots) {
        if (idx < snapshot.size()) {
            sum += mode[idx] * snapshot[idx];
        }
    }
    sum /= static_cast<float>(snapshots.size());

    // Add tiny random noise (simulating model uncertainty)
    return sum * ML_LEARNING_RATE + dist_(rng_);
}

float MLEnhancer::computeNorm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}