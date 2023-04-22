import csv
import re
from logging import *


class CsvReader():

    def __init__(self, filename):
        with open(filename) as data_file:
            self.content = list(csv.reader(data_file, delimiter=';'))
        getLogger(__name__).info("read %s content", filename)

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
        getLogger(__name__).info('read %d lines', len(str_samples))
        return str_samples

    def get_value(self, row, column):
        value = re.sub(r'[^a-zA-Z0-9.,]', "", self.content[row][column])
        value = value.replace(',', '.')
        return value if (value != "") & (value != "?") else None
