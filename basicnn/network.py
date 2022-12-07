import numpy as np


# Math Functions


def sigmoid(x):
    return 1/(1 + np.e ** -x)


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


class Network:

    def __init__(self, layer_sizes):
        self._wabs = []
        self.input_layer_size = layer_sizes[0]
        self.output_layer_size = layer_sizes[-1]
        self.num_act_layers = len(layer_sizes)
        for i in range(len(layer_sizes) - 1):
            weights = np.random.rand(layer_sizes[i + 1], layer_sizes[i] + 1)
            self._wabs.append(weights)

    def feedforward(self, input_acts):
        if len(input_acts) != self.input_layer_size:
            err = "Input activations should match input layer size of "
            raise ValueError(err + str(self.input_layer_size))

        acts = [np.array(input_acts)]
        for layer in self._wabs:
            acts_1 = np.append(acts[-1], 1)
            next_acts = sigmoid(layer.dot(acts_1))
            acts.append(next_acts)

        return acts

    def _d_cost_act(self, layer, idx, expected_out, acts):
        if layer == self.num_act_layers - 1:
            return 2 * (acts[layer][idx] - expected_out[idx])

            # j = 0, 1, ... (length next layer)
        accum_partial_cost_act = 0
        for out_neur in range(0, len(acts[layer + 1])):
            partial_cost_act = self._d_cost_act(layer + 1, out_neur, expected_out, acts)

            sum = self._wabs[layer][out_neur].dot(np.append(acts[layer], 1))
            partial_act_sum = d_sigmoid(sum)

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

    # assume train_batch is np array of shape (num samples, input_layer_size)
    # assume train_labels is an (np) array of shape (num samples, )
    # assume len(train_batch) == len(train_labels)
    def backprop(self, train_batch, train_labels):
        grad = [np.zeros(np.shape(layer)) for layer in self._wabs]

        # iterate over batch
        for i in range(len(train_labels)):
            # [0, 0, ..., 1, 0, 0 ...], 1 at label index
            expected_out = [0] * self.output_layer_size
            expected_out[train_labels[i]] = 1

            acts = self.feedforward(train_batch[i])

            # edge matrices
            for layer in reversed(range(0, len(grad))):
                # row in matrix
                layer_shape = np.shape(grad[layer])
                for out_neur in range(0, layer_shape[0]):
                    # col in matrix (last col is biases)
                    for in_neur in range(0, layer_shape[1] - 1):
                        change = self._d_cost_weight(
                            layer, out_neur, in_neur, expected_out, acts)
                        grad[layer][out_neur, in_neur] += change

                    grad[layer][out_neur, -1] = self._d_cost_bias(
                        layer, out_neur, expected_out, acts)

        for layer in range(0, len(grad)):
            # Average change to cost over all examples and
            # apply gradient descent
            self._wabs[layer] -= grad[layer] / len(train_labels)
