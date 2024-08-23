print("===============================================")
print("MNIST Maximum Feasible Ball Experiment")
print("===============================================")
import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

from nn_verifier.NN_verifier import NN_verifier
from problem_classes.digit_recognition import DigitRecognition
from utils.train_neural_network import train_neural_network
from utils.NeuralNet import NeuralNet
import torch
from utils.export_model_to_onnx import export_model_to_onnx
import pickle
import os
import numpy as np

# create digit recognition instance
digit_recognition_instance = DigitRecognition(true_label_index=4112, input_size = 50)

# Generate digit recognition data only if it doesn't already exist
train_data, train_targets, test_data, test_targets = digit_recognition_instance.input_output()

# Train neural network only if the model file doesn't already exist
n_components = 50
model = NeuralNet(input_size=n_components, output_size=10)
model_file = 'nn_models/MNIST_nn_model.pth'
pca_file = 'pca_models/MNIST_pca_model.pkl'
if not os.path.exists(model_file):
    model, pca = train_neural_network(train_data, train_targets, test_data, test_targets, model_save_name=model_file, nn_model=NeuralNet(input_size=n_components, output_size=10), criterion=torch.nn.CrossEntropyLoss(), plot=True, n_components=n_components)
else:
    print(f"Model file {model_file} already exists, skipping training.")
    model.load_state_dict(torch.load(model_file))
    with open(pca_file, 'rb') as f:
        pca = pickle.load(f)

# Export model to ONNX format
file_path = export_model_to_onnx(model)

# Get the center
digit_recognition_instance.pca = pca
center = digit_recognition_instance.get_center()

# Find the maximum feasible region for each norm type
NN_verifier_instance = NN_verifier(onnx_file=file_path, problem_class=digit_recognition_instance, nn_model=model, norm_type='infinity', epsilon_infinity=10, margin=1e-6, center=center)
B, output, _ = NN_verifier_instance.closest_point()