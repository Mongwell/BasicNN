"""Library for creating simple feed forward neural networks."""

import numpy as np
import random

# Math Functions


def sigmoid(x):
    """Math sigmoid function for normalizing activations."""
    return 1.0 / (1.0 + np.exp(-x))


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
        message = f"Dimension {dim_a} ({val_a}) should match dimension {dim_b} ({val_b})."
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
        if len(input_acts) != self.input_layer_size:
            raise DimensionError(
                "len(input_acts)",
                len(input_acts),
                "self.input_layer_size",
                self.input_layer_size,
            )

        acts = [np.array(input_acts)]
        sums = [np.array(input_acts)]
        for layer in self._wabs:
            acts_1 = np.append(acts[-1], 1)  # Allows bias addition to be part of mat-vec mult

            sum = layer.dot(acts_1)
            sums.append(sum)

            next_acts = sigmoid(sum)
            acts.append(next_acts)

        return acts, sums

    def _compute_grad_single(self, example, label):
        if len(example) != self.input_layer_size:
            raise DimensionError(
                "len(example)", len(example), "self.input_layer_size", self.input_layer_size
            )
        grad_single = [np.zeros(np.shape(layer)) for layer in self._wabs]

        # Ideal output activations should be 1 at label index, 0 otherwise
        expected_out = [0] * self.output_layer_size
        expected_out[label] = 1

        (acts, sums) = self.feedforward(example)

        # Base case, partial deriv for activation in last layer
        last_layer = 2 * (acts[-1] - expected_out)
        cost_act_partials = np.array(last_layer)

        # Iterate backwards through layers. Grad layers = Acts layers - 1.
        # Last layer of act partials already calculated, so each layer of
        # wabs matches with its corresponding input acts
        for layer in reversed(range(0, len(grad_single))):
            act_sum_partials = d_sigmoid(sums[layer + 1])

            # d cost / d biases and d cost / d weights
            cost_bias_partials = np.multiply(cost_act_partials, act_sum_partials)
            acts_1 = np.append(acts[layer], 1)
            grad_single[layer] = np.outer(cost_bias_partials, acts_1)

            # d cost / d act
            cost_act_partials = self._wabs[layer].transpose()[:-1].dot(cost_bias_partials)

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
        if len(train_batch) != len(train_labels):
            raise DimensionError(
                "len(train_batch)", len(train_batch), "len(train_labels)", len(train_labels)
            )
        grad = [np.zeros(np.shape(layer)) for layer in self._wabs]

        # iterate over batch
        for (example, label) in zip(train_batch, train_labels):
            grad_single = self._compute_grad_single(example, label)

            for layer in range(0, len(grad)):
                grad[layer] += grad_single[layer]

        # Average change to cost over batch and apply gradient descent
        for layer in range(0, len(grad)):
            self._wabs[layer] -= grad[layer] / len(train_labels)


class ClassificationModel:
    """
    Wrapper/abstraction to use Network object as a (image) classifier.

    Attributes:
        _n -- Network that will be used during training, testing, and/or inference
        class_names -- list of strings to use when making classifications
    """

    def __init__(self, nn, class_names):
        """
        Construct model with a network and class labels.

        Arguments:
            nn -- Network to use
            classes -- array of string names for output labels. Must have same
                length as output layer of nn.
        """
        if nn.output_layer_size != len(class_names):
            raise DimensionError(
                "nn.output_layer_size", nn.output_layer_size, "len(class_names)", len(class_names)
            )

        self._n = nn
        self.class_names = class_names

    def train(self, dataset, labels, batch_size, epochs):
        """
        Train the network on a dataset.

        Shuffle the data each epoch, and split into new batches before running backpropagation

        Arguments:
            dataset -- full dataset to train on as a 2d array
            labels -- integer list of labels for each example in the dataset
            batch_size -- size of each batch that the dataset will be split into. backpropagation
                will run on 1 batch instead of the entire dataset
            epochs -- number of training passes through this entire dataset
        """
        dset_shape = np.shape(dataset)
        if dset_shape[0] != len(labels):
            raise DimensionError("len(dataset)", dset_shape[0], "len(labels)", len(labels))
        if dset_shape[1] != self._n.input_layer_size:
            raise DimensionError(
                "np.shape(dataset)[1]",
                dset_shape[1],
                "_n.input_layer_size",
                self._n.input_layer_size,
            )

        labeled_dset = list(zip(dataset, labels))
        for _ in range(0, epochs):
            random.shuffle(labeled_dset)

            for batch_start in range(0, dset_shape[0], batch_size):
                batch = labeled_dset[batch_start: batch_start + batch_size]
                (batch_train, batch_labels) = zip(*batch)
                batch_train = np.array(batch_train)
                batch_labels = np.array(batch_labels)
                self._n.backprop(batch_train, batch_labels)

    def test(self, dataset, labels):
        """
        Test the accuracy of the network.

        Arguments:
            dataset -- dataset to test on as a 2d array
            labels -- integer list of labels for each examples in the dataset
        """
        dset_shape = np.shape(dataset)
        if dset_shape[0] != len(labels):
            raise DimensionError("len(dataset)", dset_shape[0], "len(labels)", len(labels))
        if dset_shape[1] != self._n.input_layer_size:
            raise DimensionError(
                "np.shape(dataset)[1]",
                dset_shape[1],
                "_n.input_layer_size",
                self._n.input_layer_size,
            )

        correct = 0
        for example, label in zip(dataset, labels):
            prediction = self._n.feedforward(example)[0][-1].argmax()
            if prediction == label:
                correct += 1

        accuracy = correct / dset_shape[0]
        return accuracy

    def infer(self, data):
        """
        Predict the class of a single piece of data.

        Arguments:
            data -- data to classify as vector of input activations
        """
        if len(data) != self._n.input_layer_size:
            raise DimensionError(
                "len(data)", len(data), "_n.input_layer_size", self._n.input_layer_size
            )

        prediction = self._n.feedforward(data)[0][-1].argmax()
        return (prediction, self.class_names[prediction])
