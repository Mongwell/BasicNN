"""Library for creating simple feed forward neural networks."""

import numpy as np

# Math Functions


def sigmoid(x):
    """Math sigmoid function for normalizing activations."""
    return 1.0/(1.0 + np.exp(-x))


def d_sigmoid(x):
    """Fast first derivative of above sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)


class DimensionError(Exception):
    """Exception type for matrix/vector dimension or length mismatch when expected to be equal."""

    def __init__(self, dim_a, val_a, dim_b, val_b):
        """
        Create an exception for two dimensions.

        Arguments:
            dim_a -- (string) name of first dimension
            val_a -- value of first dimension
            dim_b -- (string) name of second dimension
            val_b -- value of second dimension
        """
        message = "Dimension {} ({}) should match dimension {} ({})."
        message.format(dim_a, val_a, dim_b, val_b)
        super().__init__(message)


class Network:
    """
    Neural network object.

    Attributes:
        _wabs -- array of matrices containing weights and biases where the
            value in matrix k row i col j is the weight of the input activation
            j for the output activation i in layer k. The last column in each
            matrix are the biases for each output activation i.

        input_layer_size -- number of activations in the input layer

        output_layer_size -- number of activations in the output layer

        num_act_layers -- number of activation vectors for a forward pass
            through the network (including input and output layers). One greater
            than the number of matrices in _wabs
    """

    def __init__(self, layer_sizes):
        """
        Construct the network and initialize with randoms weights and biases in the range (-1, 1).

        Arguments:
            layer_sizes -- array of lengths for the activation vectors
        """
        self._wabs = []
        self.input_layer_size = layer_sizes[0]
        self.output_layer_size = layer_sizes[-1]
        self.num_act_layers = len(layer_sizes)

        # Initialize weights and biases randomly
        for i in range(len(layer_sizes) - 1):
            weights = np.random.rand(layer_sizes[i + 1], layer_sizes[i] + 1)
            weights = (weights * 2) - 0.9999999999999999
            self._wabs.append(weights)

    def feedforward(self, input_acts):
        """
        Forward pass through the network.

        Returns all activations (including input data)

        Arguments:
            input_acts -- input data for the network
        """
        # TODO: replace this error with a DimensionError
        if len(input_acts) != self.input_layer_size:
            err = "Input activations should match input layer size of "
            raise ValueError(err + str(self.input_layer_size))

        acts = [np.array(input_acts)]
        for layer in self._wabs:
            acts_1 = np.append(acts[-1], 1)  # Allows bias addition to be part of mat-vec mult
            next_acts = sigmoid(layer.dot(acts_1))
            acts.append(next_acts)

        return acts

    def _d_cost_act(self, layer, idx, expected_out, acts):
        # Base case, partial deriv for activation in last layer
        if layer == self.num_act_layers - 1:
            return 2 * (acts[layer][idx] - expected_out[idx])

        # Need to add up effect of this activation on all activations in next layer
        accum_partial_cost_act = 0
        for out_neur in range(0, len(acts[layer + 1])):
            partial_cost_act = self._d_cost_act(layer + 1, out_neur, expected_out, acts)

            # Next activation layer's weights (layer + 1) are in _wabs[layer+1-1] since
            # layers of matrices is one less than activation layers
            sum = self._wabs[layer][out_neur].dot(np.append(acts[layer], 1))
            partial_act_sum = d_sigmoid(sum)

            # Same note above applies here.
            partial_sum_act = self._wabs[layer][out_neur, idx]

            accum_partial_cost_act += partial_cost_act * partial_act_sum * partial_sum_act

        return accum_partial_cost_act

    def _d_cost_weight(self, layer, out_neur, in_neur, expected_out, acts):
        partial_sum_weight = acts[layer][in_neur]
        partial_cost_sum = self._d_cost_bias(layer, out_neur, expected_out, acts)

        return partial_cost_sum * partial_sum_weight

    def _d_cost_bias(self, layer, out_neur, expected_out, acts):
        partial_cost_act = self._d_cost_act(layer + 1, out_neur, expected_out, acts)

        sum = self._wabs[layer][out_neur].dot(np.append(acts[layer], 1))
        partial_act_sum = d_sigmoid(sum)

        partial_sum_bias = 1

        return partial_cost_act * partial_act_sum * partial_sum_bias

    def _compute_grad_single(self, example, label):
        # TODO: add check with DimensionError for len(example) == self.input_layer_size
        grad_single = [np.zeros(np.shape(layer)) for layer in self._wabs]

        # Ideal output activations should be 1 at label index, 0 otherwise
        expected_out = [0] * self.output_layer_size
        expected_out[label] = 1

        acts = self.feedforward(example)

        # Iterate backwards through layers over each weight and bias
        for layer in reversed(range(0, len(grad_single))):
            layer_shape = np.shape(grad_single[layer])
            for out_neur in range(0, layer_shape[0]):
                for in_neur in range(0, layer_shape[1] - 1):  # last col is biases
                    partial_cost_weight = self._d_cost_weight(
                        layer, out_neur, in_neur, expected_out, acts)
                    grad_single[layer][out_neur, in_neur] += partial_cost_weight

                grad_single[layer][out_neur, -1] = self._d_cost_bias(
                    layer, out_neur, expected_out, acts)

        return grad_single

    def backprop(self, train_batch, train_labels):
        """
        Run backpropagation algorithm on a batch of input data.

        Computes individual gradients (partial derivatives for each weight and bias) for each
        example, then takes averages over batch. Updates weights and biases once.

        Arguments:
            train_batch -- array of input data
            train_labels -- array of labels for input data

        Lengths of train_batch and train_labels must match
        """
        # TODO: add check with DimensionError for train_batch and train_labels lengths
        grad = [np.zeros(np.shape(layer)) for layer in self._wabs]

        # iterate over batch
        for (example, label) in zip(train_batch, train_labels):
            grad_single = self._compute_grad_single(example, label)

            for layer in range(0, len(grad)):
                grad[layer] += grad_single[layer]

        # Average change to cost over batch and apply gradient descent
        for layer in range(0, len(grad)):
            self._wabs[layer] -= grad[layer] / len(train_labels)
