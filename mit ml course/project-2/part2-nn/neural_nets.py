import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    return max(0,x)

def rectified_linear_unit_derivative(x):
    if(x > 0):
        return 1
    else:
        return 0

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1', dtype=np.float64)
        self.hidden_to_output_weights = np.matrix('1 1 1', dtype=np.float64)
        self.biases = np.matrix('0; 0; 0', dtype=np.float64)
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points =[((5, -8), -3), ((0, 2), 2), ((3, 2), 5), ((8, 6), 14), ((5, -7), -2), ((0, -9), -9), ((-5, 9), 4), ((3, -10), -7), ((-5, -2), -7), ((-3, 7), 4)]
        #[((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):  
        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]], dtype=np.float64) # 2 by 1
 
        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = (self.input_to_hidden_weights @ input_values) + (self.biases)
        
        relu = np.vectorize(rectified_linear_unit)
        layer_act = np.vectorize(output_layer_activation)
        dlayer_act = np.vectorize(output_layer_activation_derivative)
        
        hidden_layer_activation = relu(hidden_layer_weighted_input)
        output =  self.hidden_to_output_weights @ hidden_layer_activation
        
        activated_output = layer_act(output)
        ### Backpropagation ###

        # Compute gradients
        output_layer_error = activated_output-y
        hidden_layer_error = np.multiply(dlayer_act(activated_output),np.transpose(self.hidden_to_output_weights))*output_layer_error

        relu_deriv = np.vectorize(rectified_linear_unit_derivative)

        bias_gradients = np.multiply(hidden_layer_error,relu_deriv(hidden_layer_weighted_input))
        hidden_to_output_weight_gradients = np.transpose(np.multiply(hidden_layer_activation, output_layer_error))
        input_to_hidden_weight_gradients = bias_gradients @ np.transpose(input_values)
        
        # Use gradients to adjust weights and biases using gradient descent
        self.biases -= self.learning_rate*bias_gradients
        self.input_to_hidden_weights -= self.learning_rate*input_to_hidden_weight_gradients
        self.hidden_to_output_weights -= self.learning_rate*hidden_to_output_weight_gradients
        
    def predict(self, x1, x2):
        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = (self.input_to_hidden_weights @ input_values) + (self.biases)
        
        relu = np.vectorize(rectified_linear_unit)
        layer_act = np.vectorize(output_layer_activation)
        
        hidden_layer_activation = relu(hidden_layer_weighted_input)
        output =  self.hidden_to_output_weights @ hidden_layer_activation
        activated_output = layer_act(output)

        return activated_output.item()
    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):
        print("bias grad: " + str(self.biases))
        print("in->out grad: " + str(self.input_to_hidden_weights))
        print("hid->out: grad" + str(self.hidden_to_output_weights))
        print()
        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)
            print("bias grad: " + str(self.biases))
            print("in->out grad: " + str(self.input_to_hidden_weights))
            print("hid->out: grad" + str(self.hidden_to_output_weights))
            print()

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):
        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
#x.test_neural_network()
