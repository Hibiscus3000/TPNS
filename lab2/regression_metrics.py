from metrics import *


def mae(result, expected):
    N = len(result)
    sum = 0
    for i in range(0, N):
        if expected[i] is not None:
            sum += abs(result[i] - expected[i])
        else:
            N -= 1
    return sum / N


def mse(result, expected):
    N = len(result)
    sum = 0
    for i in range(0, N):
        if expected[i] is not None:
            sum += (result[i] - expected[i]) ** 2
        else:
            N -= 1
    return sum / N


def rmse(result, expected):
    return math.sqrt(mse(result, expected))


def r2(result, expected):
    sample_mean = get_sample_mean(expected)
    numerator = 0
    denominator = 0
    N = len(result)
    for i in range(0, N):
        if expected[i] is not None:
            numerator += (result[i] - expected[i]) ** 2
            denominator += (expected[i] - sample_mean) ** 2

    return 1 - numerator / denominator


def mape(result, expected):
    numerator = 0
    denominator = 0
    N = len(result)

    for i in range(0, N):
        if expected[i] is not None:
            numerator += abs(result[i] - expected[i])
            denominator += abs(expected[i])
        else:
            N -= 1

    return numerator / denominator / N
