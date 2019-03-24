import numpy as np


def rmse(actual, predicted):
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = np.sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score
