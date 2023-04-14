import abc


class Reader(abc.ABC):

    def get_values(self, start_row, end_row, start_column, end_column):
        str_samples = {}
        for i in range(start_row, end_row + 1):
            str_samples[i] = []
            sample_present = False
            for j in range(start_column, end_column + 1):
                value = self.get_value(i, j)
                str_samples[i].append(value)
                if value is not None:
                    sample_present = True
            if sample_present is False:
                del str_samples[i]
        return str_samples

    @abc.abstractmethod
    def get_value(self, row, column):
        """get "column" attribute from "row" sample"""