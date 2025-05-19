print("===============================================")
print("CIFAR-10 LEVIS Beta Experiment")
print("===============================================")

import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

from nn_verifier.NN_verifier import NN_verifier
from problem_classes.image_classfication import ImageClassificationProblem
from utils.train_neural_network import train_neural_network
from utils.NeuralNet import NeuralNet
from utils.export_model_to_onnx import export_model_to_onnx
import torch
import pickle
import os
import numpy as np

# ========== User options ==========
use_pca = True   
pca_components = 50
true_label_index = 1000  # valid for CIFAR-10 test set

n_components = pca_components if use_pca else None

# ========== Create CIFAR-10 problem instance ==========
problem_instance = ImageClassificationProblem(
    dataset_name='cifar10',
    true_label_index=true_label_index,
    pca=None  # will be set later
)

# ========== Load data ==========
train_data, train_targets, test_data, test_targets = problem_instance.input_output()
input_size = n_components if use_pca else problem_instance.input_size

# ========== Model setup ==========
model = NeuralNet(input_size=input_size, output_size=problem_instance.output_size)
model_file = f'nn_models/CIFAR10_nn_model_{"pca" if use_pca else "nopca"}_{input_size}.pth'
pca_file = f'pca_models/CIFAR10_pca_model_{input_size}.pkl'

# ========== Train or Load Model ==========
if not os.path.exists(model_file):
    print("Training neural network...")
    model, pca = train_neural_network(
        train_data=train_data,
        train_target=train_targets,
        test_data=test_data,
        test_target=test_targets,
        model_save_name=model_file,
        nn_model=NeuralNet(input_size=input_size, output_size=problem_instance.output_size),
        criterion=torch.nn.CrossEntropyLoss(),
        plot=True,
        n_components=n_components
    )
    if use_pca:
        os.makedirs(os.path.dirname(pca_file), exist_ok=True)
        with open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
else:
    print(f"Model file {model_file} already exists, skipping training.")
    model.load_state_dict(torch.load(model_file))
    if use_pca:
        with open(pca_file, 'rb') as f:
            pca = pickle.load(f)
    else:
        pca = None

# ========== Set PCA in problem instance ==========
problem_instance.pca = pca

# ========== Export to ONNX ==========
onnx_path = export_model_to_onnx(model)

# ========== Get center ==========
center = problem_instance.get_center()

# ========== Apply LEVIS-Beta ==========
NN_verifier_instance = NN_verifier(
    onnx_file=onnx_path,
    problem_class=problem_instance,
    nn_model=model,
    norm_type='infinity',
    epsilon_infinity=10,
    margin=1e-6,
    center=center
)

regions = NN_verifier_instance.LEVIS_beta(directions=1, epsilon=1e-4)
print(f"The radii are: {list(regions.values())}")
