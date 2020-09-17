import numpy as np

class NeuralNetwork:

    def __init__(self, layer_sizes, weights, biases):
        self.weights = weights
        self.biases = biases

        self.num_layers = len(layer_sizes)
        self.activations = [np.zeros((size, 1)) for size in layer_sizes]

    def feedforward(self, inputs):
        """
        Feed a sample through the network,
        return the prediction/output
        """

        # the input layer is the first activation layer
        self.activations[0] = inputs

        # forward propagation
        for i in range(self.num_layers - 1):
            # z is 'weighted input'
            z = np.matmul(self.weights[i], self.activations[i].reshape(-1,1)) + self.biases[i]
            self.activations[i + 1] = self.sigmoid(z)

        # the output layer is the final activation layer
        return self.activations[-1]
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))