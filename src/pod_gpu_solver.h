// pod_gpu_solver.h
#ifndef POD_GPU_SOLVER_H
#define POD_GPU_SOLVER_H

#include <vector>

class PODGPUSolver {
public:
    PODGPUSolver(int gridX, int gridY, int numSnapshots);

    void computePOD(const std::vector<std::vector<float>>& snapshots);
    std::vector<std::vector<float>> getModes() const;
    std::vector<float> getSingularValues() const;

private:
    int gridX_;
    int gridY_;
    int numSnapshots_;
    std::vector<std::vector<float>> podModes_;
    std::vector<float> singularValues_;
};

#endif // POD_GPU_SOLVER_H