import numpy as np
import pyomo.environ as pyo
from omlt import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
import torch 
from copy import deepcopy
import itertools


import numpy as np

class NN_verifier:
    """
    """

    def __init__(self, onnx_file, problem_class, center, nn_model=None, norm_type = 'infinity', epsilon_infinity=10, margin=1e-3):
        """
        Attributes:
        onnx_file (str): Path to the ONNX model file.

        nn_model: The neural network model object, default is None.
        constraint_function: Function to add the constraints to the Pyomo model.

        epsilon_infinity (float): Radius for the infinity norm constraints around the center.
        center (np.array): Central point in the input space around which bounds are calculated.
        original_center (np.array): Original central point in the input space.
        margin (float): Small margin used in constraints to ensure non-strict inequalities.

        input_bounds (dict): Calculated bounds for the input variables.
        network_definition: Loaded ONNX network definition.
        formulation: OMLT formulation created from the ONNX model.

        model: The Pyomo model object.
        norm_type (str): Type of norm used ('infinity', 'l1', or 'l2') for the optimization problem.
        """
        self.onnx_file = onnx_file
        self.problem_class = problem_class
        self.nn_model = nn_model

        self.original_center = center
        self.reset_center()

        self.epsilon_infinity = epsilon_infinity
        self.margin = margin

        self.input_size = problem_class.input_size
        self.output_size = problem_class.output_size

        self.input_bounds = self._calculate_bounds()
        self.load_onnx()

        self.model = None    
        self.norm_type = norm_type  # default norm type
        self.ord = self._norm_ord_converter()
    
    ##################################################### SETUP ################################################

    def reset_center(self):
        """
        Reset the center to the original center.
        """
        self.center = deepcopy(self.original_center)

    def _norm_ord_converter(self):
        """
        Convert the norm type to the ord value used in numpy.
        """
        if self.norm_type == 'infinity':
            return np.inf
        elif self.norm_type == 'l1':
            return 1
        elif self.norm_type == 'l2':
            return 2
        else:
            raise ValueError("Invalid norm type.")

    def _calculate_bounds(self):
        """Calculates the input bounds based on the center and epsilon_infinity."""
        lb = self.center - self.epsilon_infinity
        ub = self.center + self.epsilon_infinity
        print(f"Input bounds: {lb} - {ub}")
        return {i: (float(lb[i]), float(ub[i])) for i in range(len(self.center))}

    def load_onnx(self):
        """
        Loads the ONNX model and sets up the neural network formulation.

        This method writes the ONNX model with the specified input bounds, 
        loads the network definition, and creates the neural network formulation.
        """
        write_onnx_model_with_bounds(self.onnx_file, None, self.input_bounds)
        self.network_definition = load_onnx_neural_network_with_bounds(self.onnx_file)
        self.formulation = FullSpaceNNFormulation(self.network_definition)
    

    #################################################### VARIABLES ####################################################

    def declare_variables(self):
        """
        Declares variables in the Pyomo model based on the neural network definition.

        This method sets up a concrete model with input and output variables, auxiliary variables based on the norm type,
        and constraints to link these variables with the neural network inputs and outputs.
        """
        m = pyo.ConcreteModel()
        m.nn = OmltBlock()
        m.nn.build_formulation(self.formulation)

        # Create input and output variables
        m.input = pyo.Var(range(self.input_size), domain=pyo.Reals)
        m.output = pyo.Var(range(self.output_size), domain=pyo.Reals)

        # Create auxiliary variable depending on the norm type
        if self.norm_type == 'infinity':
            m.auxiliary = pyo.Var(domain=pyo.NonNegativeReals)
        elif self.norm_type == 'l1':
            m.auxiliary = pyo.Var(range(self.input_size), domain=pyo.NonNegativeReals)

        # Constraints to connect neural network inputs and outputs
        m.connect_inputs = pyo.ConstraintList()
        m.connect_outputs = pyo.ConstraintList()
        for i in range(self.input_size):
            m.connect_inputs.add(m.input[i] == m.nn.inputs[i])
        for j in range(self.output_size):
            m.connect_outputs.add(m.output[j] == m.nn.outputs[j])

        self.model = m

    #################################################### CONSTRAINTS ####################################################

    
    def collinearity_constraints(self, instance, b):
        """
        Define collinearity constraints based on the selected norm type and set up the objective function.

        Args:
            instance (pyo.ConcreteModel): The Pyomo model.
            b (np.array): The bad point we want to be collinear with.
        """

        # collinearity constraint
        instance.k = pyo.Var(domain=pyo.NonNegativeReals)
        instance.collinear = pyo.ConstraintList()
        for i in range(self.input_size):
            instance.collinear.add(instance.input[i] == self.center[i] - instance.k * (b[i] - self.center[i])) # b is B[-1]
    
        return instance


    def orthogonality_constraints(self, instance, B):
        """
        Define orthogonality constraints based on the selected norm type and set up the objective function.

        Args:
            instance (pyo.ConcreteModel): The Pyomo model.
            B (np.array): The bad points we want to be orthogonal to.
        """

        instance.orthogonal = pyo.ConstraintList()
        for i in range(len(B)):
            if i%2 == 0:
                instance.orthogonal.add(sum((instance.input[j] - self.center[j]) * (B[i][j] - self.center[j]) for j in range(self.input_size)) == 0)

        return instance


    def norm_constraints(self, input_neurons, m):
        """
        Define norm constraints based on the selected norm type and set up the objective function.

        Args:
            input_neurons (int): Number of input neurons.
            m (pyo.ConcreteModel): The Pyomo model.
        """
        instance = m.create_instance()
        if self.norm_type == 'l1':
            instance.l1_constraints = pyo.ConstraintList()
            for i in range(input_neurons):
                instance.l1_constraints.add(instance.input[i] - self.center[i] <= instance.auxiliary[i])
                instance.l1_constraints.add(-instance.auxiliary[i] <= instance.input[i] - self.center[i])
            instance.obj = pyo.Objective(expr=sum(instance.auxiliary[i] for i in range(input_neurons)), sense=pyo.minimize)
        elif self.norm_type == 'l2':
            instance.obj = pyo.Objective(expr=sum((instance.input[i] - self.center[i])**2 for i in range(input_neurons)), sense=pyo.minimize)
        else:  # 'infinity' norm by default
            instance.l1_constraints = pyo.ConstraintList()
            for i in range(input_neurons):
                instance.l1_constraints.add(instance.input[i] - self.center[i] <= instance.auxiliary)
                instance.l1_constraints.add(-instance.auxiliary <= instance.input[i] - self.center[i])
            instance.obj = pyo.Objective(expr=instance.auxiliary, sense=pyo.minimize)
        return instance


    def create_opt_problem(self, index = 0, orthogonality = False, collinearity = False, B = None):
        """
        Creates the optimization problem based on the selected norm type and constraints.

        Args:
            index (int): Index of the constraint configuration to apply.
            orthogonality (bool): Whether to apply orthogonality constraints.
            collinearity (bool): Whether to apply collinearity constraints.
            B (np.array): The bad points to be orthogonal to."""
        
        self._assert_opt_problem(orthogonality, collinearity, B)

        instance = self.norm_constraints(self.input_size, self.model)

        if collinearity:
            instance = self.collinearity_constraints(instance, B[-1])
        if orthogonality:
            instance = self.orthogonality_constraints(instance, B)
    
        instance = self.problem_class.add_constraint(instance, index = index, margin = self.margin, center = self.center)

        return instance

    
    def _assert_opt_problem(self, orthogonality, collinearity, B):

        """"
        The conditions that must be met:
        - At most one of collinearity and orthogonality can be True
        - If one is True, size of B must be non-zero
        """

        if collinearity and orthogonality:
            raise ValueError("At most one of collinearity and orthogonality can be True.")
        if collinearity or orthogonality:
            if B is None:
                raise ValueError("If collinearity or orthogonality is True, B must be non-empty.")
            if len(B) == 0:
                raise ValueError("If collinearity or orthogonality is True, B must be non-empty.")
    

    ##################################################################### SOLVE ################################################################


    def solve_opt_problem(self, instance = None, solver = None, B = None):
        """
        Solves the optimization problem using the CBC solver and returns the model's inputs and outputs.

        Prints the outcome of the optimization process. If the solver reaches an optimal solution,
        it reports success; otherwise, it indicates failure.

        Returns:
            list: Values of the input variables after solving the optimization problem.
            list: Values of the output variables after solving the optimization problem.
        """

        if B is None:
            B = []

        # assert that in l2 case, solver is bonmin
        if solver is None:
            if self.norm_type == 'l2':
                solver = 'bonmin'
            else:
                solver = 'cbc'
        
        if instance == None: 
            instance = self.model

        if self.norm_type == 'l2':
            assert solver == 'bonmin', "Solver must be 'bonmin' for l2 norm."

        solver_status = pyo.SolverFactory(solver).solve(instance, tee=False)  # tee=False suppresses solver output

        # Initialize lists to store input and output values
        inputs = []
        outputs = []

        B_matrix = deepcopy(B)

        # Check if the solver found an optimal solution
        if solver_status.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("Problem solved successfully.")
            # Retrieve and store input values
            inputs = [instance.input[i].value for i in instance.input]
            # Retrieve and store output values
            outputs = [instance.output[i].value for i in instance.output]
            display = True
        else:
            print("The solver failed to solve the problem.")
            display = False
        
        if self.norm_type == 'infinity' or self.norm_type == 'l1':
            norm_value = instance.obj()
        else:
            norm_value = np.sqrt(instance.obj())

        B_matrix.append(inputs)

        return B_matrix, outputs, norm_value, display
    
    
    def closest_point(self, index = 0, orthogonality = False, collinearity = False, B = None, render = True):

        self.declare_variables()
        instance = self.create_opt_problem(index, orthogonality, collinearity, B)
        B, output, norm_value, display = self.solve_opt_problem(instance, B=B)

        if display and render:
            self.display_results(B[-1], output, self.center, self.norm_type)
        
        return B, output, norm_value


    def maximum_feasible_ball(self, orthogonality = False, collinearity = False, B = None, render = False):
        """
        Determines the configuration that results in the smallest feasible region by varying constraint settings.

        Iterates over a predefined set of configurations, each differing in the constraints applied,
        to find which configuration yields the smallest objective value, indicating the tightest feasible region.

        Returns:
            float: The size of the smallest feasible region found.
            dict: The configuration that resulted in the smallest feasible region.
        """
        smallest_region = float('inf')
        best_config = None
        best_input = None
        best_output = None
    
        if B is None:
            B = []

        for index in range(self.problem_class.num_constraints):
            # Create and solve the optimization problem with the current configuration
            self.declare_variables()
            instance = self.create_opt_problem(index, orthogonality, collinearity, B)
            B_matrix, output, _, display = self.solve_opt_problem(instance = instance, B=B)
            input = B_matrix[-1]

            if not display:
                continue 
                
            # Fetch the objective function value which quantifies the size of the feasible region
            if self.norm_type == 'infinity' or self.norm_type == 'l1':
                region_size = instance.obj() if instance.obj.expr() is not None else float('inf')
            else:
                region_size = np.sqrt(instance.obj()) if instance.obj.expr() is not None else float('inf')

            # Check if the current configuration results in a smaller feasible region
            if region_size < smallest_region:
                smallest_region = region_size
                best_config = index
                best_input = input
                best_output = output

        # print(f"Smallest feasible region size: {smallest_region}")
        # print(f"Best configuration: {self.problem_class.constraint_description[best_config]}")
        
        if render == True:
            self.display_results(best_input, best_output)
            self.render(best_input, best_output, self.center, self.norm_type, smallest_region)

        B.append(best_input)
        return B, output, smallest_region
    

    ##################################################################### DISPLAY #####################################################################


    def display_results(self, inputs = None, outputs = None, center = None, norm_type = None):
        """
        Displays the results of the optimization problem, including input and output values.

        Args:
            inputs (list): Values of the input variables.
            outputs (list): Values of the output variables.
            center (list): Central point in the input space.
        """
        if inputs is None:
            inputs = [self.model.input[i].value for i in self.model.input]
        if outputs is None:
            outputs = [self.model.output[i].value for i in self.model.output]
        if center is None:
            center = self.center
        if norm_type is None:
            norm_type = self.norm_type

        self.problem_class.display_results(inputs, outputs, center, norm_type)
    
    def render(self, inputs = None, outputs = None, center = None, norm_type = None, norm_value = None):
        if inputs is None:
            inputs = [self.model.input[i].value for i in self.model.input]
        if outputs is None:
            outputs = [self.model.output[i].value for i in self.model.output]
        if center is None:
            center = self.center
        if norm_type is None:
            norm_type = self.norm_type
        if norm_value is None:
            if norm_type == 'infinity' or norm_type == 'l1':
                norm_value = self.model.obj()
            else:
                norm_value = np.sqrt(self.model.obj())

        self.problem_class.render(inputs, outputs, center, norm_type, norm_value)
        
    
    ##################################################################### LEVIS-ALPHA #####################################################################


    def LEVIS_alpha(self, epsilon = 1):
        """
        Applies the LEVIS-alpha algorithm to find a large verifiable region.
        """

        self.reset_center()

        B, _, radius = self.maximum_feasible_ball()
        center_list = [self.center] # store the centers
        radii = [radius] # store the radii
        len_list = []

        iteration = 0

        while (len(radii) < 2 or abs(radii[-1] - radii[-2]) > epsilon):
            # solve the collinear point problem
            B, _, radius = self.maximum_feasible_ball(collinearity=True, B = B)

            # for (input_size - 1) times, solve the orthogonal point problem and collinear point problem
            for i in range(self.input_size - 1):
                B, _, radius = self.maximum_feasible_ball(orthogonality=True, B = B)
                B, _, radius = self.maximum_feasible_ball(collinearity=True, B = B)

            self.center, len_mean = self.problem_class.compute_new_center(B, self.nn_model)
            # self.center = np.mean(B, axis = 0)

            center_list.append(self.center)
            # len_list.append(len_mean)

            B = []
            B, _, radius = self.maximum_feasible_ball()
            radii.append(radius)

            iteration += 1
            if len(radii) > 1:
                print(f"Radius difference: {abs(radii[-1] - radii[-2])}")

        return center_list, radii, len_list 
    

    ##################################################################### LEVIS-BETA #####################################################################
    
    def midpoint_finder(self, c,r,y):
        """
        Find the midpoint of the line segment between the boundary of the region formed by (c,r) and y.

        Args:
            c (np.array): Center of the region.
            r (float): Radius of the region.
            y (np.array): Possibly a bad point.
        """

        c_y = c - y
        c_y_norm = np.linalg.norm(c_y, ord = self.ord)
        x = c - r * c_y/c_y_norm

        return (x + y)/2
    
    def is_point_inside_any_region(self, point, regions):
        """
        Check if a point is inside any region defined by a dictionary of centers and radii based on a specified norm.
        
        Parameters:
        - point: Tuple or list representing the coordinates of the point (e.g., [x, y]).
        - regions: Dictionary where keys are region names and values are tuples (center, radius).
        - norm_type: Type of norm to use ('l1', 'l2', or 'linf').

        Returns:
        - True if the point is inside any region, False otherwise.
        """

        point = np.array(point)

        for center, radius in regions.items():
            center = np.array(center)
            
            distance = np.linalg.norm(point - center, ord=self.ord)
            
            if distance <= radius:
                return True

        return False
    

    def LEVIS_beta(self, epsilon = 0.1, maximum_ball = False, directions = None):
        """
        Applies the LEVIS-beta algorithm to find a large verifiable region.

        Args:
        - epsilon (float): Minimum distance between centers.
        - maximum_ball (bool): If True, the algorithm uses the method to find the maximum feasible ball.
        - all_directions (int): Number of directions to consider for the orthogonal problem.

        Note: for DC-OPF, the maximum ball is True while for digit recognition the maximum ball is False.
        """

        if maximum_ball:
            method = self.maximum_feasible_ball
        else:
            method = self.closest_point

        if directions is None:
            directions = self.input_size - 1

        self.reset_center()

        centers = [self.center]
        regions = {}

        while (len(centers)!=0):
            self.center = centers.pop()
            B, _, radius = method(render = False)
            b_i = B[0]
            if radius <= epsilon:
                continue
                
            regions[tuple(self.center)] = radius
            for i in range(directions):

                B, _, _ = method(orthogonality=True, B=B, render = False)
                B.append(B[-1])
                print("============================")
                print(B[-1])
                print("=========================")
                b_i_plus_1 = B[-1]
                midpoint = self.midpoint_finder(self.center, radius, b_i_plus_1)

                distance_to_b_i_plus_1 = np.linalg.norm(b_i_plus_1 - midpoint, ord=self.ord)
                
                while (self.is_point_inside_any_region(midpoint, regions)) and (distance_to_b_i_plus_1 > epsilon):
                    midpoint = (midpoint + b_i_plus_1) / 2
                    distance_to_b_i_plus_1 = np.linalg.norm(b_i_plus_1 - midpoint, ord=self.ord)
                    if self.is_point_inside_any_region(midpoint, regions):
                        print("Midpoint is inside a region.")
                    elif distance_to_b_i_plus_1 <= epsilon:
                        print("Distance is less than epsilon.")
                
                if distance_to_b_i_plus_1 <= epsilon:
                    print(distance_to_b_i_plus_1)
                    continue
                else:
                    centers.append(midpoint)
            
            print("The length of the centers is ", len(centers))
        
        return regions
    
    ########################################################################################################################################################



