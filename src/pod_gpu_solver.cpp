// pod_gpu_solver.cpp
#include "pod_gpu_solver.h"
#include "utils.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

PODGPUSolver::PODGPUSolver(int gridX, int gridY, int numSnapshots)
    : gridX_(gridX), gridY_(gridY), numSnapshots_(numSnapshots) {}

void PODGPUSolver::computePOD(const std::vector<std::vector<float>>& snapshots) {
    int m = gridX_ * gridY_;
    int n = numSnapshots_;
    int lda = m;
    
    std::cout << "[PODGPUSolver] Allocating GPU memory..." << std::endl;

    // Flatten snapshots into a single column-major matrix
    std::vector<float> h_data(m * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            h_data[j + i * m] = snapshots[i][j];
        }
    }

    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));

    float* d_U = nullptr; // Left singular vectors (POD modes)
    float* d_S = nullptr; // Singular values
    float* d_VT = nullptr; // Right singular vectors (transpose)

    CUDA_CHECK(cudaMalloc(&d_U, m * m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, std::min(m, n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_VT, n * n * sizeof(float)));

    cusolverDnHandle_t cusolverH = nullptr;
    CUBLAS_CHECK(cusolverDnCreate(&cusolverH));

    int work_size = 0;
    float* d_work = nullptr;
    int* devInfo = nullptr;
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));

    // Query working space size for SVD
    CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(cusolverH, m, n, &work_size));
    CUDA_CHECK(cudaMalloc(&d_work, work_size * sizeof(float)));

    signed char jobu = 'A'; // All columns of U
    signed char jobvt = 'A'; // All rows of V^T

    std::vector<float> h_S(std::min(m, n));
    std::vector<float> h_U(m * m);
    std::vector<float> h_VT(n * n);

    std::cout << "[PODGPUSolver] Performing SVD on GPU..." << std::endl;

    CUSOLVER_CHECK(cusolverDnSgesvd(
        cusolverH, jobu, jobvt,
        m, n,
        d_data, lda,
        d_S,
        d_U, lda,
        d_VT, n,
        d_work, work_size,
        nullptr, // No device workspace
        devInfo));

    int dev_info_h = 0;
    CUDA_CHECK(cudaMemcpy(&dev_info_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (dev_info_h != 0) {
        throw std::runtime_error("[PODGPUSolver] SVD failed to converge!");
    }

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, std::min(m, n) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_U.data(), d_U, m * m * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "[PODGPUSolver] Processing POD modes..." << std::endl;

    // Reshape POD modes and singular values
    podModes_.resize(numSnapshots_);
    for (int modeIdx = 0; modeIdx < numSnapshots_; ++modeIdx) {
        podModes_[modeIdx].resize(gridX_ * gridY_);
        for (int i = 0; i < gridX_ * gridY_; ++i) {
            podModes_[modeIdx][i] = h_U[i + modeIdx * m];
        }
    }

    singularValues_ = h_S;

    // Free memory
    cudaFree(d_data);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_VT);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);

    std::cout << "[PODGPUSolver] POD computation complete." << std::endl;
}

std::vector<std::vector<float>> PODGPUSolver::getModes() const {
    return podModes_;
}

std::vector<float> PODGPUSolver::getSingularValues() const {
    return singularValues_;
}