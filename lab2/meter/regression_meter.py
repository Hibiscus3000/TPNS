from meter import Meter
from regression_metrics import *


class RegressionMeter(Meter):

    def __init__(self, number_of_targets, key_meter='R2', rises_to_stop=10):
        super().__init__(key_meter, rises_to_stop, number_of_targets)

    def count_for_selection(self, result, expected, selection_dictionary):
        selection_dictionary["MAE"].append(mae(result, expected))
        selection_dictionary["MSE"].append(mse(result, expected))
        selection_dictionary["RMSE"].append(rmse(result, expected))
        selection_dictionary["R2"].append(r2(result, expected))
        selection_dictionary["MAPE"].append(mape(result, expected))

    def get_initial_metrics(self):
        return {"MAE": [], "MSE": [], "RMSE": [], "R2": [], "MAPE": []}

    def check_metric_worsened(self, current, previous):
        if 'R2' != self.key_metric:
            return current > previous
        else:
            return current < previous
