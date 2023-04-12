import abc

class Reader(abc.ABC):

    def get_values(self, start_row, end_row, start_column, end_column):
        values = []
        for i in range(start_row, end_row):
            values.append([])
            for j in range(start_column, end_column):
                values[i - start_row].append(self.get_value(i, j))
        return values

    @abc.abstractmethod
    def get_value(self, row, column):
        """get "column" attribute from "row" sample"""
