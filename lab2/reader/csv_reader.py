from reader.reader import Reader
from logging import *
import csv
import re


class CsvReader(Reader):

    def read_file(self, file):
        return list(csv.reader(file, delimiter=';'))

    def get_value(self, row, column):
        value = re.sub(r'[^0-9.,]', "", self.content[row][column])
        value = value.replace(',', '.')
        return value if value != "" else None
