import numpy as np


# Math Functions


def sigmoid(x):
    return 1/(1 + np.e ** -x)


class Network:

    def __init__(self, layer_sizes):
        self._wabs = []
        self.input_layer_size = layer_sizes[0]
        self.output_layer_size = layer_sizes[-1]
        for i in range(len(layer_sizes) - 1):
            weights = np.random.rand(layer_sizes[i + 1], layer_sizes[i] + 1)
            self._wabs.append(weights)

    def feedforward(self, input_acts):
        if len(input_acts) != self.input_layer_size:
            err = "Input activations should match input layer size of "
            raise ValueError(err + str(self.input_layer_size))

        acts = [np.array(input_acts)]
        for layer in self._wabs:
            acts_1col = np.append(acts[-1], 1)
            next_acts = sigmoid(layer.dot(acts_1col))
            acts.append(next_acts)

        return acts
