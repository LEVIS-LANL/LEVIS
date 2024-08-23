print("===============================================")
print("DC-OPF LEVIS Alpha Experiment")
print("===============================================")
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

import os
import logging
import torch
import numpy as np
from problem_classes.DC_OPF import DC_OPF
from utils.train_neural_network import train_neural_network
from utils.export_model_to_onnx import export_model_to_onnx
from sklearn.model_selection import train_test_split
from utils.NeuralNet import NeuralNet
from nn_verifier.NN_verifier import NN_verifier
from utils.plotting_norms import plot_3d_norm_balls


# File paths
data_file = 'datasets/DC_OPF_data.csv'
model_file = 'nn_models/DC_OPF_nn_model.pth'
system_data_file = 'system_data.ieee9'

# Set the logging level for Pyomo to CRITICAL to suppress warnings
logging.getLogger('pyomo').setLevel(logging.CRITICAL)

# create DC_OPF instance
dc_opf = DC_OPF(system_data_file)

# Generate DC OPF data only if it doesn't already exist
if not os.path.exists(data_file):
    dc_opf.generate_data(1000, data_file)
else:
    print(f"Data file {data_file} already exists, skipping data generation.")

X,y = dc_opf.input_output(data_file)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = NeuralNet()

# Train neural network only if the model file doesn't already exist
if not os.path.exists(model_file):
    model,_ = train_neural_network(
    X_train, y_train, X_test, y_test, model_save_name=model_file)
else:
    print(f"Model file {model_file} already exists, skipping training.")
    model.load_state_dict(torch.load(model_file))


# Export model to ONNX format
file_path = export_model_to_onnx(model)


# create the center
expanded_center = dc_opf.demand
center = np.zeros(dc_opf.input_size)
for i, demand_bus in enumerate(dc_opf.demand_buses):
    center[i] = expanded_center[demand_bus]

norm_type = 'infinity'
NN_verifier_instance = NN_verifier(onnx_file = file_path, problem_class=dc_opf, nn_model=model, norm_type = norm_type, epsilon_infinity=1000, margin=1e-3, center = center)

# Apply LEVIS-Alpha
print("LEVIS-Alpha")
centers, radii, _ = NN_verifier_instance.LEVIS_alpha()
print(f"The final center is {centers[-1]} with radius {radii[-1]}")