from abc import ABC, abstractmethod
import numpy as np

class ProblemClass(ABC):
    """
    Abstract base class for problem classes.
    """

    @abstractmethod
    def generate_data(self):
        """
        Generates the necessary data for the machine learning task.
        """
        raise NotImplementedError("This method must be overridden by the subclass")

    @abstractmethod
    def input_output(self):
        """
        Returns the input and output data and targets of the machine learning task.
        """
        raise NotImplementedError("This method must be overridden by the subclass")

    @abstractmethod
    def add_constraint(self, instance, index, margin, center):
        """
        Adds constraints to the optimization model based on the problem specifics.
        """
        raise NotImplementedError("This method must be overridden by the subclass")

    @abstractmethod
    def _constraint_description(self):
        """
        Returns the description of the constraints for the task.
        """
        raise NotImplementedError("This method must be overridden by the subclass")

    @abstractmethod
    def compute_new_center(self, B, nn_model):
        """
        Computes the new center based on the given parameters.
        """
        raise NotImplementedError("This method must be overridden by the subclass")

    @abstractmethod
    def display_results(self, inputs, outputs, center, norm_type):
        """
        Displays the results of the machine learning task.
        """
        raise NotImplementedError("This method must be overridden by the subclass")

    def norm_calculator(self, input_vector, center, norm_type):
        """
        Calculates the norm of the difference between a given input vector and the center.
        
        Parameters:
        - input_vector (list or np.array): The input vector for which the norm needs to be calculated.
        - center (list or np.array): The center vector to calculate the norm against.
        - norm_type (str): The type of norm to calculate ('l1', 'l2', or 'infinity'). 

        Returns:
        - float: The calculated norm.
        """
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
