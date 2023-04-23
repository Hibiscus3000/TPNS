import abc
from logging import *


class Meter(abc.ABC):

    def __init__(self, key_metric, deterioration_to_stop, number_of_targets):
        self.learning = []
        self.predicting = []
        for i in range(0, number_of_targets):
            self.learning.append(self.get_initial_metrics())
            self.predicting.append(self.get_initial_metrics())
        self.key_metric = key_metric
        self.number_of_targets = number_of_targets
        self.deterioration_to_stop = deterioration_to_stop
        self.learning_deterioration = [0] * number_of_targets
        self.predicting_deterioration = [0] * number_of_targets

    def count_metrics(self, result_learning, expected_learning, result_predicting, expected_predicting):
        for i in range(0, self.number_of_targets):
            self.count_for_selection(result_learning[i], expected_learning[i], self.learning[i])
            self.count_for_selection(result_predicting[i], expected_predicting[i], self.predicting[i])

            if self.check_stop(self.learning[i], 'learning', self.learning_deterioration, i):
                return self.learning_deterioration[i]
            if self.check_stop(self.predicting[i], 'predicting', self.predicting_deterioration, i):
                return self.predicting_deterioration[i]

        return 0

    @abc.abstractmethod
    def count_for_selection(self, result, expected, selection_dictionary):
        pass

    def log_last_metrics(self):
        getLogger(__name__).info("_____________________________LEARNING_____________________________")
        for metrix_name, metric in self.learning.items():
            getLogger(__name__).info(f'{metrix_name}: {metric:.2f}')
        getLogger(__name__).info("____________________________PREDICTING____________________________")

    @abc.abstractmethod
    def get_initial_metrics(self):
        pass

    def check_stop(self, selection_dictionary, selection_name, det_counters, target_id):
        key_metric = selection_dictionary[self.key_metric]
        N = len(key_metric) - 1
        if N < 1:
            return False
        if self.check_metric_worsened(key_metric[N], key_metric[N - 1]):
            det_counters[target_id] += 1
        else:
            det_counters[target_id] = 0
        if det_counters[target_id] >= self.deterioration_to_stop:
            if 'y' != input(f'{self.key_metric} deteriorate {det_counters[target_id]} times in a row '
                            f'for {selection_name} selection, would you like to continue?'):
                return True
            else:
                return False
        return False
