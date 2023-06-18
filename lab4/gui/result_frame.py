import abc
import random

from tkinter import *

import reader

font_family = 'Comic Sans'

class ResultFrame(Frame, abc.ABC):

    @abc.abstractclassmethod
    def show_result(self, result):
        # expected encoded result
        pass

    @abc.abstractclassmethod
    def clear(self):
        pass

class NumeralFrame(ResultFrame):

    def __init__(self, window):
        super().__init__(window)
        self.label = Label(self, text = '', font=(font_family, 30, 'bold'))
        self.phrases = ('Oh, i know! It\'s {}!', 'It\'s {}', 'I think it\'s {}...',
                        'You drew {}', 'I believe it\'s {}', 'I see {}')
        self.label.pack()

    def show_result(self, result):
        if result is not None:
            self.label.config(text = self.phrases[random.randrange(len(self.phrases))]
                              .format(reader.decode(result)))

    def clear(self):
        self.label.config(text = '')

class CertantyFrame(ResultFrame):

    def __init__(self, window):
        super().__init__(window)
        self.std_font_size = 16
        self.labels = [Label(self, text=f'{i}: ', font=(font_family, self.std_font_size))
                       for i in range(0,10)]
        for label in self.labels:
            label.pack()

    def show_result(self, result):
        if result is not None:
            for i in range(0, len(self.labels)):
                label = self.labels[i]
                label.config(text = f'{i}: ' + self.get_certanty(result[i]))

    @abc.abstractclassmethod
    def get_certanty(self, result):
        pass

    def clear(self):
        for i in range(0, len(self.labels)):
            label = self.labels[i]
            label.config(text = f'{i}: ')

class CertantyInNumbersFrame(CertantyFrame):

    def get_certanty(self, result):
        return '{:.3f}'.format(result)

class CertantyInPorcentageFrame(CertantyFrame):

    def get_certanty(self, result):
        return '{:.1f}'.format(100 * result) + '%'