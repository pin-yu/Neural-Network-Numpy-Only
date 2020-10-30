import numpy as np


def random_unbias_weight(dim1, dim2):
    # The weight should be independent to node numbers. Therefore, multiply
    # randn by np.sqrt(1./dim1).
    # * is element-wise multiplication
    return np.random.randn(dim1, dim2) * np.sqrt(1. / dim1)


def random_unbias_bias(dim):
    return np.zeros((1, dim))


class NeuralNetwork:
    def __init__(self, input_layer_dim, hidden_layer_dim,
                 output_layer_dim, activation, output_activation):
        self.input_layer_dim = input_layer_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.output_layer_dim = output_layer_dim

        self.activation = activation
        self.output_activation = output_activation

        self._initialize_weight()
        self.back_prop = {}
        self.grads = {}

    def _initialize_weight(self):
        self.params = {
            'W1': random_unbias_weight(self.input_layer_dim, self.hidden_layer_dim),
            'b1': random_unbias_bias(self.hidden_layer_dim),
            'W2': random_unbias_weight(self.hidden_layer_dim, self.output_layer_dim),
            'b2': random_unbias_bias(self.output_layer_dim)
        }

    def forward_pass(self, X):
        batch = X.shape[0]

        self.back_prop['X'] = X

        self.back_prop['Z1'] = np.matmul(
            X, self.params['W1']) + self.params['b1']

        self.back_prop['A1'] = self.activation.forward(self.back_prop['Z1'])

        self.back_prop['Z2'] = np.matmul(
            self.back_prop['A1'],
            self.params['W2']) + self.params['b2']

        self.back_prop['A2'] = self.output_activation.forward(
            self.back_prop['Z2'])

        # ensure correctness
        assert self.back_prop['Z1'].shape == (batch, self.hidden_layer_dim)
        assert self.back_prop['A1'].shape == (batch, self.hidden_layer_dim)
        assert self.back_prop['Z2'].shape == (batch, self.output_layer_dim)
        assert self.back_prop['A2'].shape == (batch, self.output_layer_dim)

        return self.back_prop['A2']

    def backward_pass(self, y):
        batch = y.shape[0]
        norm = 1. / batch

        dZ2 = self.output_activation.backward(self.back_prop['A2'], y)
        self.grads['W2'] = norm * np.matmul(self.back_prop['A1'].T, dZ2)
        self.grads['b2'] = norm * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.matmul(dZ2, self.params['W2'].T)
        dZ1 = dA1 * self.activation.backward(self.back_prop['Z1'])
        self.grads['W1'] = norm * np.matmul(self.back_prop['X'].T, dZ1)
        self.grads['b1'] = norm * np.sum(dZ1, axis=0, keepdims=True)

        # ensure correctness
        assert dZ2.shape == (batch, self.output_layer_dim)
        assert self.grads['W2'].shape == (
            self.hidden_layer_dim, self.output_layer_dim)
        assert self.grads['b2'].shape == (1, self.output_layer_dim)

        assert dA1.shape == (batch, self.hidden_layer_dim)
        assert dA1.shape == (batch, self.hidden_layer_dim)
        assert self.grads['W1'].shape == (
            self.input_layer_dim, self.hidden_layer_dim)
        assert self.grads['b1'].shape == (1, self.hidden_layer_dim)
