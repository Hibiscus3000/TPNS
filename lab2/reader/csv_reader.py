from reader.reader import Reader
import csv

class CsvReader(Reader):
    def __init__(self, filename):
        with open(filename) as data_file:
            self.content = list(csv.reader(self.fix_nulls(data_file),
                                           delimiter=';'))

    def fix_nulls(self, csv):
        for line in csv:
            yield line.replace('\0', '')

    def get_value(self, row, column):
        value = self.content[row][column]
        return value if value != "" else None