GPU-Accelerated Proper Orthogonal Decomposition for CFD Reduced-Order Modeling
📜 Abstract
This project implements a high-performance, GPU-accelerated Proper Orthogonal Decomposition (POD) framework for reduced-order modeling (ROM) of synthetic CFD simulation data. It combines traditional POD techniques with machine learning-based post-processing to enhance mode reconstruction and noise robustness. The framework leverages CUDA programming, MPI parallelization, and modular C++ design for efficient large-scale processing.

This repository accompanies the thesis "Parallelized POD Computation for Large CFD Datasets Using GPU/MPI", submitted as part of the B.Tech in Aerospace Engineering at VIT Bhopal University.

🏗️ Project Structure
css
Copy
Edit
gpu-pod-cfd-rom/
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── main.cpp
│   ├── pod_gpu_solver.cpp / .h
│   ├── ml_enhancer.cpp / .h
│   ├── snapshot_generator.cpp / .h
│   ├── utils.cpp / .h
│   ├── config.h
├── data/
│   ├── raw_snapshots/
│   ├── processed_pod_modes/
├── experiments/
│   ├── hyperparameter_study.md
│   ├── gpu_scalability_test.md
├── notebooks/
│   ├── visualize_pod_modes.ipynb
│   ├── ml_performance_analysis.ipynb
├── results/
│   ├── figures/
│   ├── tables/
└── thesis/
    └── github_references.txt
⚙️ How to Build
Requirements:
C++17 compliant compiler (e.g., GCC 9+, Clang)

CUDA Toolkit 11.x or higher

Python 3.8+ (for visualization notebooks)

Libraries:

Eigen3 (for CPU fallback SVD)

cuBLAS/cuSOLVER (for GPU SVD)

matplotlib, numpy (Python side)

Building:
bash
Copy
Edit
mkdir build
cd build
cmake ..
make
Or, simple g++/nvcc compile (if no CMake):

bash
Copy
Edit
nvcc -std=c++17 ../src/*.cpp -o pod_gpu_rom -lcublas -lcusolver
🚀 How to Run
bash
Copy
Edit
./pod_gpu_rom
By default, it generates synthetic CFD snapshots, applies GPU-accelerated POD, enhances the modes using a lightweight ML model, and saves the reduced-order basis into /data/processed_pod_modes/.

📊 How to Visualize
Open the notebooks:

bash
Copy
Edit
cd notebooks
jupyter notebook visualize_pod_modes.ipynb
Plot leading POD modes, singular value spectrum, ML-enhanced reconstruction accuracy, etc.

🧠 Machine Learning Enhancement
The ML module (src/ml_enhancer.cpp) applies a simple feed-forward neural network (or optionally a kernel regression model) to enhance the fidelity of truncated POD reconstructions, especially under noisy or sparse data scenarios.

📈 Experiments and Benchmarking
GPU Scalability (experiments/gpu_scalability_test.md)

ML Hyperparameter Tuning (experiments/hyperparameter_study.md)

📂 Data
data/raw_snapshots/: Synthetic high-fidelity CFD snapshots (velocity fields, etc.)

data/processed_pod_modes/: Reduced POD bases and reconstruction results.

📚 References
This codebase is an implementation of concepts discussed in:

Lumley, J. L., The Structure of Inhomogeneous Turbulent Flows (1967)

Sirovich, L., Turbulence and the Dynamics of Coherent Structures (1987)

Halko et al., Randomized Algorithms for Matrix Decompositions (2011)

And extended with GPU and ML enhancements as described in [Your Thesis Title].

📝 License
This project is licensed under the MIT License.

🎯 Citation
If you find this code useful for your research, please cite:

latex
Copy
Edit
@misc{yourname2025gpurom,
  title = {GPU-Accelerated POD for CFD ROM},
  author = {Saharsh Gupta},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/gpu-pod-cfd-rom}}
}
🔥 Notes
For detailed references used in the thesis, see thesis/github_references.txt
