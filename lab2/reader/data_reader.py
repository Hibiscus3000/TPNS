from reader.reader import Reader

class DataReader(Reader):
    def __init__(self, filename):
        with open(filename,"r") as datafile:
            self.content = datafile.readlines()

    def get_value(self, row, column):
        value = self.content[row][column]
        return value if value != "?" else None