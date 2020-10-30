import time
from tqdm import trange

from src.metric import accuracy
from src.loss import cross_entropy
from src.utility import shuffle, get_batch

from src.nn import NeuralNetwork
from src.optimizer import Optimizer
from src.activation import Relu, Sigmoid, Softmax


class Model:
    def __init__(self, layer_dims, activation='relu',
                 output_activation='softmax'):
        self.layer_dims = layer_dims
        self._act(activation, output_activation)
        self.NN = NeuralNetwork(
            layer_dims[0],
            layer_dims[1],
            layer_dims[2],
            self.act,
            self.output_act)

    def _act(self, activation, output_activation):
        if activation == 'relu':
            self.act = Relu
        elif activation == 'sigmoid':
            self.act = Sigmoid
        else:
            raise ValueError(
                'We only support relu and simoid as an activation.')

        if output_activation == 'softmax':
            self.output_act = Softmax
        else:
            raise ValueError('We only support Softmax as our last activation.')

    def train(self, X_train, y_train, batch_size=32, epochs=10,
              optimizer='momentum', lr_rate=1e-4, beta=.9):
        print('start training...')

        optimizer = Optimizer(optimizer=optimizer, lr_rate=lr_rate, beta=beta)

        total_data_num = X_train.shape[0]
        batches = total_data_num // batch_size

        for epoch in range(1, epochs + 1):
            print('Epoch {}'.format(epoch))

            start_time = time.time()
            # Shuffle
            X_shuffle, y_shuffle = shuffle(X_train, y_train)

            for current_batch in trange(batches):
                X = get_batch(
                    X_shuffle,
                    current_batch,
                    batch_size,
                    total_data_num)
                y = get_batch(
                    y_shuffle,
                    current_batch,
                    batch_size,
                    total_data_num)

                # Forward pass
                out = self.NN.forward_pass(X)

                # Backward pass
                self.NN.backward_pass(y)

                # Optimize
                optimizer.update(self.NN)

            out = self.NN.forward_pass(X_shuffle)
            acc = accuracy(out, y_shuffle)
            loss = cross_entropy(out, y_shuffle)
            end_time = time.time() - start_time

            print(
                "time elapse={:.2f}s, train acc={:.2f}, train loss={:.2f}\n".format(
                    end_time,
                    acc,
                    loss))

    def test(self, X):
        return self.NN.forward_pass(X)
