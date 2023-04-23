import math


def accuracy(confusion_matrix):
    return confusion_matrix[0] + confusion_matrix[1] / \
        (confusion_matrix[0] + confusion_matrix[1] + confusion_matrix[2] + confusion_matrix[3])


def confusion_matrix(result, expected, threshold=0.5):
    # TP, TN, FP, FN
    confusion_matrix = [0, 0, 0, 0]
    for i in range(0, len(result)):
        if expected[i] is None:
            continue
        res = 1 if result[i] >= threshold else 0
        if expected[i] == res:
            truthfulness = 2
        else:
            truthfulness = 0
        confusion_matrix[truthfulness + 1 - expected[i]] += 1

    return confusion_matrix


def precision(confusion_matrix):
    denominator = confusion_matrix[0] + confusion_matrix[1]
    return confusion_matrix[0] / denominator if 0 != denominator else -1


def recall(confusion_matrix):
    denominator = confusion_matrix[0] + confusion_matrix[3]
    return confusion_matrix[0] / denominator if 0 != denominator else -1


def f1(confusion_matrix):
    denominator = (confusion_matrix[0] + 0.5 * (confusion_matrix[0] + confusion_matrix[3]))
    return confusion_matrix[0] / denominator if 0 != denominator else -1


def false_positive_rate(confusion_matrix):
    return confusion_matrix[2] / (confusion_matrix[2] + confusion_matrix[1])


def roc(result, expected, steps):
    step = 1 / steps
    tpr = []
    fpr = []
    for i in range(0, steps):
        conf_matrix = confusion_matrix(result, expected, i * step)
        tpr.append(recall(conf_matrix))
        fpr.append(false_positive_rate(conf_matrix))
    return tpr, fpr


def area_under_curve(result, expected):
    N = len(result)
    numerator = 0
    denominator = 0
    for i in range(0, N):
        for j in range(0, N):
            i_yi_yj = identifier(result[i], result[j])
            numerator += i_yi_yj * identifier_der(expected[i], expected[j])
            denominator += i_yi_yj

    return numerator / denominator


def identifier(y1, y2):
    return y1 < y2


def identifier_der(y1, y2):
    return 0.5 if math.isclose(y1, y2) else y1 < y2
