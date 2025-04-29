// snapshot_generator.h
#ifndef SNAPSHOT_GENERATOR_H
#define SNAPSHOT_GENERATOR_H

#include <vector>

class SnapshotGenerator {
public:
    SnapshotGenerator(int gridX, int gridY, int numSnapshots);

    std::vector<std::vector<float>> generateSnapshots();

private:
    int gridX_;
    int gridY_;
    int numSnapshots_;
};

#endif // SNAPSHOT_GENERATOR_H