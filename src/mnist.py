import os
import pandas as pd

from src.utility import one_hot_encode

from sklearn.datasets import fetch_openml


class Mnist:
    def __init__(self):
        self.mnist_data = './dataset/mnist_784_data.csv'
        self.mnist_label = './dataset/mnist_784_label.csv'
        self._fetch()

    def _fetch(self):
        print('fetching mnist dataset')
        if os.path.isfile(self.mnist_data) and os.path.isfile(
                self.mnist_label):
            self.X, self.y = self._load_csv()
        else:
            self.X, self.y = self._download_mnist()
            self._save_csv()

    def get(self):
        return self.X, self.y

    def normalization(self):
        self.X = self.X / 255
        return self

    def one_hot_encode(self):
        self.y = one_hot_encode(self.y, 10)
        return self

    def _load_csv(self):
        X = pd.read_csv(self.mnist_data, header=None).to_numpy()
        y = pd.read_csv(self.mnist_label, header=None).astype(
            'str').to_numpy().squeeze()
        return X, y

    def _download_mnist(self):
        mnist = fetch_openml('mnist_784')
        X = mnist['data']
        y = mnist['target']
        return X, y

    def _save_csv(self):
        pd.DataFrame(self.X).to_csv(self.mnist_data, header=False, index=False)
        pd.DataFrame(
            self.y).to_csv(
            self.mnist_label,
            header=False,
            index=False)
