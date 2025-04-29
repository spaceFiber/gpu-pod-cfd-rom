// main.cpp
#include <iostream>
#include "config.h"
#include "snapshot_generator.h"
#include "pod_gpu_solver.h"
#include "ml_enhancer.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    try {
        std::cout << "=============================================" << std::endl;
        std::cout << "      GPU-Accelerated POD for CFD ROM         " << std::endl;
        std::cout << "=============================================" << std::endl;

        Timer global_timer;
        global_timer.start();

        // Step 1: Generate synthetic CFD snapshots
        std::cout << "[INFO] Generating synthetic CFD snapshots..." << std::endl;
        SnapshotGenerator snapshotGen(GRID_SIZE_X, GRID_SIZE_Y, NUM_SNAPSHOTS);
        auto snapshots = snapshotGen.generateSnapshots();
        std::cout << "[INFO] Snapshot generation completed." << std::endl;

        // Step 2: Perform GPU-accelerated POD
        std::cout << "[INFO] Performing GPU-accelerated POD decomposition..." << std::endl;
        PODGPUSolver podSolver(GRID_SIZE_X, GRID_SIZE_Y, NUM_SNAPSHOTS);
        podSolver.computePOD(snapshots);
        auto podModes = podSolver.getModes();
        auto singularValues = podSolver.getSingularValues();
        std::cout << "[INFO] POD decomposition completed." << std::endl;

        // Step 3: Apply Machine Learning enhancement (optional)
        if (ENABLE_ML_ENHANCEMENT) {
            std::cout << "[INFO] Applying Machine Learning enhancement on POD modes..." << std::endl;
            MLEnhancer mlEnhancer;
            podModes = mlEnhancer.enhanceModes(podModes, snapshots);
            std::cout << "[INFO] ML enhancement completed." << std::endl;
        }

        // Step 4: Save results
        std::cout << "[INFO] Saving POD modes and singular values..." << std::endl;
        saveModesToDisk(podModes, OUTPUT_MODES_FOLDER);
        saveSingularValuesToDisk(singularValues, OUTPUT_SINGULAR_VALUES_FILE);
        std::cout << "[INFO] Results saved successfully." << std::endl;

        global_timer.stop();
        std::cout << "=============================================" << std::endl;
        std::cout << "        Total execution time: " 
                  << global_timer.elapsedMilliseconds() << " ms" << std::endl;
        std::cout << "=============================================" << std::endl;
    }
    catch (const std::exception& ex) {
        std::cerr << "[ERROR] Exception occurred: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "[ERROR] Unknown error occurred." << std::endl;
        return EXIT_FAILURE;
    }

    #include <fstream>

void saveVectorToBinary(const std::vector<float>& vec, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }
    file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    file.close();
}

    return EXIT_SUCCESS;
}