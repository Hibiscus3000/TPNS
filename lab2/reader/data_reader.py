from reader.reader import Reader

class DataReader(Reader):

    def read_file(self, file):
        return file.readlines()

    def get_value(self, row, column):
        value = self.content[row][column]
        return value if value != "?" else None