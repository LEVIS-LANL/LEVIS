# LEVIS: Large Exact Verifiable Input Spaces for Neural Networks

## Introduction
LEVIS (Large Exact Verifiable Input Spaces) is a novel framework designed to enhance the robustness of neural networks in safety-critical applications. This repository contains the implementation of the LEVIS framework, which includes LEVIS-α and LEVIS-β algorithms. For more details, refer to our paper on [arXiv](https://www.arxiv.org/pdf/2408.08824).

## Repository Structure
- `datasets/`: Contains dataset files like `DC_OPF_data.csv` used for training and testing.
- `experiments/`: Scripts for running various experiments, such as `MNIST_LEVIS_beta.py` and `DC_OPF_LEVIS_alpha.py`.
- `nn_models/`: Pre-trained neural network models like `MNIST_nn_model.pth`.
- `nn_verifier/`: Contains the neural network verification scripts like `NN_verifier.py`.
- `pca_models/`: PCA models for dimensionality reduction, e.g., `MNIST_pca_model.pkl`.
- `problem_classes/`: Python classes defining problem-specific settings like `DC_OPF.py`.
- `system_data/`: System-specific data files such as `ieee9.py`.
- `utils/`: Utility scripts including `train_neural_network.py` and `export_model_to_onnx.py`.

## Installation

```bash
git clone https://github.com/your-repository/LEVIS.git
cd LEVIS
pip install -r requirements.txt


