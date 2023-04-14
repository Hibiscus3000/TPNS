import abc

class Reader(abc.ABC):

    def get_values(self, start_row, end_row, start_column, end_column):
        str_samples = {}
        for i in range(start_row, end_row + 1):
            str_samples[i] = []
            for j in range(start_column, end_column + 1):
                str_samples[i].append(self.get_value(i, j))
        return str_samples

    @abc.abstractmethod
    def get_value(self, row, column):
        """get "column" attribute from "row" sample"""
