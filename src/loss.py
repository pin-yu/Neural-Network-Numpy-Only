import numpy as np


def cross_entropy(output, y):
    # add small values to avoid log(0)
    nozero_output = np.add(output, 1e-6)
    entropy = np.log(nozero_output)
    cross_entropy = -np.sum(np.multiply(y, entropy), axis=1, keepdims=True)

    # return mean entropy of this batch
    return np.mean(cross_entropy)
