from meter import Meter
from qualification_metrics import *


class QualificationMeter(Meter):

    def __init__(self, number_of_targets, key_meter='f1', rises_to_stop=3):
        super().__init__(key_meter, rises_to_stop, number_of_targets)

    def count_for_selection(self, result, expected, selection_dictionary):
        conf_matrix = confusion_matrix(result, expected)
        selection_dictionary["accuracy"].append(accuracy(conf_matrix))
        prec = precision(conf_matrix)
        if prec > 0:
            selection_dictionary["precision"].append(prec)
        rec = recall(conf_matrix)
        if rec > 0:
            selection_dictionary["recall"].append(rec)
        F1 = f1(conf_matrix)
        if F1 > 0:
            selection_dictionary["f1"].append(F1)
        # selection_dictionary["AUC"].append(area_under_curve(result, expected))

    def get_initial_metrics(self):
        return {"accuracy": [], "precision": [], "recall": [], "f1": [], "AUC": []}

    def check_metric_worsened(self, current, previous):
        return current < previous
