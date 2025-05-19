import torch
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from problem_classes.problem_class import ProblemClass
import pyomo.environ as pyo


class ImageClassificationProblem(ProblemClass):

    def __init__(self,
                 dataset_name='mnist',
                 train_data=None,
                 train_targets=None,
                 test_data=None,
                 test_targets=None,
                 true_label_index=0,
                 all_labels=True,
                 pca=None):

        self.dataset_name = dataset_name.lower()
        self.true_label_index = true_label_index
        self.all_labels = all_labels
        self.pca = pca

        self.train_data, self.train_targets, self.test_data, self.test_targets = self.generate_data(
            dataset_name=self.dataset_name,
            train_data=train_data,
            train_targets=train_targets,
            test_data=test_data,
            test_targets=test_targets
        )

        # Determine input size from the (possibly PCA-transformed) center
        center = self.get_center()
        self.input_size = center.shape[0]
        self.output_size = len(torch.unique(self.train_targets))
        self.true_label = self._get_true_label()
        self.constraint_description = self._constraint_description()

    ###########################################
    # Data Loading and Normalization
    ###########################################

    def generate_data(self, dataset_name,
                      train_data=None, train_targets=None,
                      test_data=None, test_targets=None):

        if train_data is None or test_data is None:
            if dataset_name == 'mnist':
                transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
                print("Loading MNIST dataset...")
                train_ds = MNIST(root='./data', train=True, download=True, transform=transform)
                test_ds = MNIST(root='./data', train=False, download=True, transform=transform)

                train_data = train_ds.data.view(len(train_ds), -1).float()
                test_data = test_ds.data.view(len(test_ds), -1).float()
                train_targets = train_ds.targets
                test_targets = test_ds.targets

            elif dataset_name == 'cifar10':
                transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                print("Loading CIFAR-10 dataset...")
                train_ds = CIFAR10(root='./data', train=True, download=True, transform=transform)
                test_ds = CIFAR10(root='./data', train=False, download=True, transform=transform)

                train_data = torch.tensor(train_ds.data).permute(0, 3, 1, 2).float()
                test_data = torch.tensor(test_ds.data).permute(0, 3, 1, 2).float()
                train_data = train_data.view(len(train_data), -1)
                test_data = test_data.view(len(test_data), -1)
                train_targets = torch.tensor(train_ds.targets)
                test_targets = torch.tensor(test_ds.targets)

            else:
                raise ValueError(f"Dataset '{dataset_name}' not supported.")
        else:
            train_data = torch.tensor(train_data).float() if not isinstance(train_data, torch.Tensor) else train_data.float()
            test_data = torch.tensor(test_data).float() if not isinstance(test_data, torch.Tensor) else test_data.float()
            train_targets = torch.tensor(train_targets) if not isinstance(train_targets, torch.Tensor) else train_targets
            test_targets = torch.tensor(test_targets) if not isinstance(test_targets, torch.Tensor) else test_targets

            if train_data.ndim > 2:
                train_data = train_data.view(len(train_data), -1)
                test_data = test_data.view(len(test_data), -1)

        # Normalize to [-1, 1]
        if train_data.max() > 1.0:
            train_data = train_data / 255.0
            test_data = test_data / 255.0
        train_data = train_data * 2 - 1
        test_data = test_data * 2 - 1

        return train_data, train_targets, test_data, test_targets

    ###########################################
    # Setup
    ###########################################

    def _get_true_label(self):
        return self.test_targets[self.true_label_index].item()

    def get_center(self):
        image = self.test_data[self.true_label_index].numpy()

        if self.pca is not None:
            image_pca = self.pca.transform([image])
            self.input_size = image_pca.shape[1]
            return image_pca.reshape(-1)
        else:
            return image

    def input_output(self):
        return self.train_data, self.train_targets, self.test_data, self.test_targets

    ###################################################### CONSTRAINTS ######################################################

    def _constraint_description(self):
        constraint_description = {}
        if self.all_labels:
            constraint_description[0] = f"Classification problem with true label {self.true_label} and all other labels."
        else:
            for i in range(self.output_size):
                constraint_description[i] = f"Classification problem with true label {self.true_label} and label {i}."
        return constraint_description

    def add_constraint(self, instance, index, margin, center):
        if self.all_labels:
            print(f"{self.constraint_description[0]}")

            M = 100
            instance.z = pyo.Var(range(self.output_size), domain=pyo.Binary)
            instance.output_property = pyo.ConstraintList()

            for i in range(self.output_size):
                if i == self.true_label:
                    continue
                instance.output_property.add(
                    instance.output[self.true_label] - instance.output[i] + margin <= M * (1 - instance.z[i])
                )

            instance.output_property.add(sum(instance.z[i] for i in range(self.output_size)) == 1)
            instance.output_property.add(instance.z[self.true_label] == 0)

        else:
            print(f"{self.constraint_description[index]}")
            assert 0 <= index < self.output_size and index != self.true_label
            instance.output_property = pyo.Constraint(
                expr=instance.output[self.true_label] - instance.output[index] + margin <= 0
            )
        return instance

    ###################################################### LEVIS ALPHA HELPER ######################################################

    def compute_new_center(self, B, nn_model):
        assert isinstance(B, np.ndarray)
        assert isinstance(nn_model, torch.nn.Module)
        return np.mean(B, axis=0), len(B)

    ###################################################### DISPLAY ######################################################

    def norm_calculator(self, input_vector, center, norm_type):
        diff = np.array(input_vector) - np.array(center)
        if norm_type == 'l1':
            return np.linalg.norm(diff, 1)
        elif norm_type == 'l2':
            return np.linalg.norm(diff, 2)
        elif norm_type == 'infinity':
            return np.linalg.norm(diff, np.inf)
        else:
            raise ValueError("Norm type must be one of: 'l1', 'l2', 'infinity'.")

    def display_results(self, inputs=None, outputs=None, center=None, norm_type=None):
        assert outputs is not None and norm_type is not None

        if center is not None and inputs is not None:
            distance = self.norm_calculator(inputs, center, norm_type)
            print(f"Distance between input and center ({norm_type}-norm): {distance:.4f}\n")

        max_output = np.max(outputs)
        for i, val in enumerate(outputs):
            print(f"Output for label {i}: {val:.4f}", end=" ")
            if i == self.true_label:
                print("(True label)", end="")
            if val == max_output:
                print(" (Predicted label)", end="")
            print()
