// snapshot_generator.cpp
#include "snapshot_generator.h"
#include <cmath>
#include <random>
#include <iostream>

SnapshotGenerator::SnapshotGenerator(int gridX, int gridY, int numSnapshots)
    : gridX_(gridX), gridY_(gridY), numSnapshots_(numSnapshots) {}

std::vector<std::vector<float>> SnapshotGenerator::generateSnapshots() {
    std::cout << "[SnapshotGenerator] Generating synthetic CFD snapshots..." << std::endl;

    std::vector<std::vector<float>> snapshots(numSnapshots_, std::vector<float>(gridX_ * gridY_, 0.0f));

    // Random generators for noise and parameter variations
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> amplitude_dist(0.8f, 1.2f);
    std::uniform_real_distribution<float> frequency_dist(0.8f, 1.2f);
    std::uniform_real_distribution<float> phase_dist(0.0f, 2.0f * M_PI);
    std::uniform_real_distribution<float> noise_dist(-0.02f, 0.02f);

    for (int snapIdx = 0; snapIdx < numSnapshots_; ++snapIdx) {
        float amplitude = amplitude_dist(rng);
        float freqX = frequency_dist(rng);
        float freqY = frequency_dist(rng);
        float phaseX = phase_dist(rng);
        float phaseY = phase_dist(rng);

        for (int iy = 0; iy < gridY_; ++iy) {
            for (int ix = 0; ix < gridX_; ++ix) {
                float x = static_cast<float>(ix) / gridX_;
                float y = static_cast<float>(iy) / gridY_;

                // Simulate synthetic vortex-like structures
                float value = amplitude * std::sin(2.0f * M_PI * freqX * x + phaseX) * std::cos(2.0f * M_PI * freqY * y + phaseY);
                value += noise_dist(rng); // Add random noise

                snapshots[snapIdx][iy * gridX_ + ix] = value;
            }
        }
    }

    std::cout << "[SnapshotGenerator] Finished generating " << numSnapshots_ << " snapshots." << std::endl;
    return snapshots;
}