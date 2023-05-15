from logging import *

import numpy as np

number_of_classes = 10

def accuracy(result, expected):
    # expected decoded results
    assert expected.shape == result.shape
    N = len(expected)
    correct = 0
    for i in range(0, N):
        if expected[i] == result[i]:
            correct += 1
    return correct / N

def confusion_matrix(result, expected, class_value):
    # expected decoded results
    assert result.shape == expected.shape 
    # TP, TN, FN, FP
    confusion_matrix = [0, 0, 0, 0]
    for i in range(0, len(result)):
        if result[i] == expected[i] == class_value:
            confusion_matrix[0] += 1
        if (result[i] != expected[i]) & (expected[i] == class_value):
            confusion_matrix[2] += 1
        if (result[i] == expected[i]) & (expected[i] != class_value):
            confusion_matrix[1] += 1
        if (result[i] != expected[i]) & (result[i] == class_value):
            confusion_matrix[3] += 1
    return confusion_matrix


def confusion_matrix_tr(result, expected, class_value, threshold):
    # expected encoded results
    assert result.shape[0] == expected.shape[0] 
    # TP, TN, FN, FP
    confusion_matrix = [0, 0, 0, 0]
    for i in range(0, len(result)):
        if result[i][class_value] > threshold:
            if class_value == expected[i]:
                confusion_matrix[0] += 1
            else:
                confusion_matrix[3] += 1
        else:
            if class_value == expected[i]:
                confusion_matrix[2] += 1
            else:
                confusion_matrix[1] += 1
    return confusion_matrix

def recall(confusion_matrix):
    denominator = (confusion_matrix[0] + confusion_matrix[2])
    return confusion_matrix[0] / denominator if 0 != denominator else -1

def precision(confusion_matrix):
    denominator = (confusion_matrix[0] + confusion_matrix[3])
    return confusion_matrix[0] / denominator if 0 != denominator else -1

def false_positive_rate(confusion_matrix):
    denominator = confusion_matrix[1] + confusion_matrix[3]
    return confusion_matrix[3] / denominator if 0 != denominator else -1

def f1(confusion_matrix):
    denominator = (confusion_matrix[0] + 0.5 * (confusion_matrix[2] + confusion_matrix[3]))
    return confusion_matrix[0] / denominator if 0 != denominator else -1

def roc(result, expected, class_value, steps = 100):
    step = 1 / (steps - 1)
    tpr = []
    fpr = []
    for i in range(0, steps):
        conf_matrix = confusion_matrix_tr(result, expected, class_value, i * step)
        true_pr = recall(conf_matrix)
        false_pr = false_positive_rate(conf_matrix)
        if (true_pr >= 0) & (false_pr >= 0):
            tpr.append(true_pr)
            fpr.append(false_pr)
    return tpr, fpr

def confusion_matrix_all(result, expected):
    confusion_matrices = []
    for i in range(0, number_of_classes):
        confusion_matrices.append(confusion_matrix(result, expected, i))
    return confusion_matrices

def metric_all(confusion_matrices, metric):
    metrics = []
    for i in range(0, number_of_classes):
        metrics.append(metric(confusion_matrices[i]))
    return metrics

def roc_all(result, expected, steps = 100):
    rocs = []
    for i in range(0, number_of_classes):
        rocs.append(roc(result, expected, i, steps))
    return rocs

def log_confusion_matrices(confusion_matrices):
    getLogger(__name__).info(f"confusion_matrices [TP, TN, FN, FP]")
    for i in range(0, len(confusion_matrices)):
        getLogger(__name__).info(f'{i} : {confusion_matrices[i]}')

def log_metrics(metrics, metric_name):
    getLogger(__name__).info(metric_name)
    for i in range(0, len(metrics)):
        getLogger(__name__).info(f'{i} : {metrics[i]}')
