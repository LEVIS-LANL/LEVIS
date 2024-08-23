import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import pyomo.environ as pyo
import itertools
from copy import deepcopy
from problem_classes.problem_class import ProblemClass


class DigitRecognition(ProblemClass):

    def __init__(self, true_label_index = 7774, input_size = 784, all_labels = True, pca = None):
        """
        Parameters
        ----------
        true_label_index(int): The index of the true label in the test dataset.
        input_size(int): The number of pixels in the image.
        all_labels(bool): A boolean indicating whether we wish to compare the true label to all other labels.
        pca_file(str): The file path to the PCA file.
        """

        self.true_label_index = true_label_index
        self.input_size = input_size
        self.pca = pca
        self.output_size = 10
        self.all_labels = all_labels
        self.constraint_description = self._constraint_description()
        self.train_dataset, self.test_dataset = self.generate_data()
        self.true_label = self._get_true_label()

    
    ###################################################### SETUP ######################################################
    
    def _get_true_label(self):
        """
        Returns the true label of the digit recognition problem.
        """
        return self.test_dataset[self.true_label_index][1]
    

    def get_center(self):
        """
        Returns the center of the digit recognition problem.
        """

        image = self.test_dataset[self.true_label_index][0]

        # Flatten the image
        image_flattened = image.view(-1).numpy()

        # Transform the image using PCA
        image_pca = self.pca.transform([image_flattened])  # 'pca' should be the PCA instance fitted on your training data

        return image_pca.reshape(-1)
        

    ###################################################### DATA GENERATION ######################################################

    def generate_data(self):
        """
        Returns the training and testing data for the digit recognition problem.
        """
        transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
        self.train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
        return self.train_dataset, self.test_dataset
    
    def input_output(self):
        """
        Returns the input and output data and target of the digit recognition problem.
        """
         # Flatten the images and normalize
        train_dataset, test_dataset = self.generate_data()
        train_data = train_dataset.data.view(len(train_dataset), -1).float() / 255.0
        test_data = test_dataset.data.view(len(test_dataset), -1).float() / 255.0
        train_targets = train_dataset.targets
        test_targets = test_dataset.targets

        return train_data, train_targets, test_data, test_targets
    

    ###################################################### CONSTRAINTS ######################################################

    def _constraint_description(self):
        """
        Returns the description of the constraint.
        """
        constraint_description = {}
        if self.all_labels:
            description = "Digit recognition problem with true label {self.true_label} and all other labels."
            constraint_description[0] = description
        else:
            for i in range(self.output_size):
                description = f"Digit recognition problem with true label {self.true_label} and label {i}."
                constraint_description[i] = description
        return constraint_description
    

    def add_constraint(self, instance, index, margin, center):
        """
        Adds constraints to the model based on the true label and all_labels parameter.
        """
        if self.all_labels:
            print(f"{self.constraint_description[0]}")

            M = 100 # big M 
            instance.z = pyo.Var(range(self.output_size), domain=pyo.Binary) # binary auxiliary variables

            instance.output_property = pyo.ConstraintList() 

            for i in range(self.output_size):
                if i == self.true_label:
                    continue
                instance.output_property.add(instance.output[self.true_label] - instance.output[i] + margin <= M * (1 - instance.z[i])) # OR constraint relaxation 
            
            instance.output_property.add(sum(instance.z[i] for i in range(self.output_size)) == 1) # only one of the 9 constraints is active 
            instance.output_property.add(instance.z[self.true_label] == 0) # the true label constraint is not active

        else:
            print(f"{self.constraint_description[index]}")

            assert 0 <= index < self.output_size, "Index out of range."
            assert index != self.true_label, "Index cannot be the true label."

            instance.output_property = pyo.Constraint(expr=instance.output[self.true_label] - instance.output[index] + margin <= 0)

        return instance
        
    
    ###################################################### LEVIS ALPHA HELPER ######################################################

    def compute_new_center(self, B, nn_model):

        """
        Computes the new center for the Levis Alpha algorithm.

        Parameters
        ----------
        B(np.ndarray): The set of bad points.
        nn_model(torch.nn.Module): The neural network model.

        The current implementation does not cover the full functionality, as LEVIS Alpha is not recommended for digit recognition problems.
        """

        assert isinstance(B, np.ndarray), "B must be a numpy array."
        assert isinstance(nn_model, torch.nn.Module), "nn_model must be a PyTorch model."

        new_center = np.mean(B, axis=0) # the new center is the mean of the points in B
        len_mean = len(B)   

        return new_center, len_mean
    
    
    ###################################################### DISPLAY ######################################################

    def norm_calculator(self, input_vector, center, norm_type):
        """
        Calculates the norm of the difference between a given input vector and the center.

        Parameters:
        - input_vector (list or np.array): The input vector for which the norm needs to be calculated.
        - center (list or np.array, optional): The center vector to calculate the norm against. Defaults to self.center.
        - norm_type (str, optional): The type of norm to calculate ('l1', 'l2', or 'infinity'). 

        Returns:
        - float: The calculated norm.
        """
        # Default to class attributes if not specified
        # Calculate the difference vector
        diff_vector = np.array(input_vector) - np.array(center)
        # Calculate the specified norm
        if norm_type == 'l1':
            norm_value = np.linalg.norm(diff_vector, ord=1)
        elif norm_type == 'l2':
            norm_value = np.linalg.norm(diff_vector, ord=2)
        elif norm_type == 'infinity':
            norm_value = np.linalg.norm(diff_vector, ord=np.inf)
        else:
            raise ValueError("Invalid norm type specified. Choose 'l1', 'l2', or 'infinity'.")

        return norm_value
    

    def display_results(self, inputs = None, outputs = None, center = None, norm_type = None):

        """
        Displays the results of the digit recognition problem.

        Parameters
        ----------
        inputs(np.ndarray): The input data.
        outputs(np.ndarray): The output data.
        center(np.ndarray): The center of the data.
        norm_type(str): The type of l_p norm to use.
        """

        assert outputs is not None, "Outputs must be provided."
        assert norm_type is not None, "Norm type must be provided."

        if center is not None and inputs is not None:
            distance = self.norm_calculator(inputs, center, norm_type)
            print(f"Distance between the input and the center: {distance}")
            print()

        max_output = np.max(outputs)
        true_output = outputs[self.true_label]

        for i in range(len(outputs)):
            print(f"Output for label {i}: {outputs[i]}", end = " ")
            if i == self.true_label:
                print(f"(True label)")
            elif outputs[i] == max_output:
                print(f"(Predicted label)")
            else:
                print()


        
    
        