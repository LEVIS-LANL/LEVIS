print("===============================================")
print("CIFAR-10 Maximum Feasible Ball Experiment")
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

# ========== User Option ==========
use_pca = True
pca_components = 50
true_label_index = 1000  # use a valid index in CIFAR-10 test set

# ========== Step 1: Create Problem Instance ==========
problem_instance = ImageClassificationProblem(
    dataset_name='cifar10',
    true_label_index=true_label_index
)

# ========== Step 2: Get Data ==========
train_data, train_targets, test_data, test_targets = problem_instance.input_output()

# ========== Step 3: Determine Input Size ==========
original_input_size = problem_instance.input_size
n_components = pca_components if use_pca else original_input_size
# If PCA is used, we need to set the input size to the number of components
# ========== Step 4: Model + File Naming ==========
model_file = f'nn_models/CIFAR10_nn_model_{"pca" if use_pca else "nopca"}_{n_components}.pth'
pca_file = f'pca_models/CIFAR10_pca_model_{n_components}.pkl'
model = NeuralNet(input_size=n_components, output_size=problem_instance.output_size)

# ========== Step 5: Train or Load Model ==========
if not os.path.exists(model_file):
    print("Training neural network...")
    model, pca = train_neural_network(
        train_data=train_data,
        train_target=train_targets,
        test_data=test_data,
        test_target=test_targets,
        model_save_name=model_file,
        nn_model=NeuralNet(input_size=n_components, output_size=problem_instance.output_size),
        criterion=torch.nn.CrossEntropyLoss(),
        plot=True,
        n_components=n_components if use_pca else None
    )
    if use_pca:
        os.makedirs(os.path.dirname(pca_file), exist_ok=True)
        with open(pca_file, 'wb') as f:
            pickle.dump(pca, f)
else:
    print(f"Model file {model_file} already exists, skipping training.")
    model.load_state_dict(torch.load(model_file))
    pca = None
    if use_pca:
        with open(pca_file, 'rb') as f:
            pca = pickle.load(f)

# ========== Step 6: Set PCA ==========
problem_instance.pca = pca

# ========== Step 7: Export to ONNX ==========
onnx_path = export_model_to_onnx(model)

# ========== Step 8: Get Center and Run Verifier ==========
center = problem_instance.get_center()

verifier = NN_verifier(
    onnx_file=onnx_path,
    problem_class=problem_instance,
    nn_model=model,
    norm_type='infinity',
    epsilon_infinity=10,
    margin=1e-6,
    center=center
)

print(f"Input size used: {problem_instance.input_size}")
B, output, _ = verifier.closest_point()
