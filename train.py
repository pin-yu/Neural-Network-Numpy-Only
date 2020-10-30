from src.mnist import Mnist
from src.model import Model

epochs = 10
batch_size = 32

# activation
activation = 'relu' # 'sigmoid'
output_activation = 'softmax'

# optimizer parameters
optimizer = 'momentum'  # 'sgd'
lr_rate = .1
beta = .9

# Please do not modify
layer_dims = [784, 64, 10]

if __name__ == '__main__':
    X, y = Mnist().normalization().one_hot_encode().get()

    model = Model(layer_dims, activation=activation, output_activation=output_activation)
    print('model initialized')

    model.train(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        optimizer=optimizer,
        lr_rate=lr_rate,
        beta=beta)
