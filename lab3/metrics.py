import math
import numpy as np

def mae(result, expected):
    return np.sum(np.abs(result - expected)) / len(result)


def mse(result, expected):
    return np.sum((result - expected) ** 2) / len(result)


def rmse(result, expected):
    return math.sqrt(mse(result, expected))


def r2(result, expected):
    sample_mean = np.average(expected)
    return 1 - np.sum((result - expected) ** 2) / np.sum((expected - sample_mean) ** 2)


def mape(result, expected):
    return np.sum(np.abs(result - expected)) / np.sum(np.abs(expected)) / len(result)