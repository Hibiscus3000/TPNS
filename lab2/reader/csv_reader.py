from reader.reader import Reader
import csv
import re

class CsvReader(Reader):
    def __init__(self, filename):
        with open(filename) as data_file:
            self.content = list(csv.reader(data_file,
                                           delimiter=';'))

    # def fix_nulls(self, csv):
    #     for line in csv:
    #         yield line.replace('\0', '')

    def get_value(self, row, column):
        value = re.sub(r'[^0-9.,]', "",self.content[row][column])
        value = value.replace(',','.')
        return value if value != "" else None