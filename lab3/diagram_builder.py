import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

def show_prediction_results(train_output, train_expected, test_output, test_expected):
    fig, axs = plt.subplots(2, 1, figsize=(20, 12), constrained_layout=True)
    add_plot(axs[0], train_output, train_expected, 'TRAIN')
    add_plot(axs[1], test_output, test_expected, 'TEST')
    fig.suptitle('Energy')
    plt.show()


def add_plot(plot, output, expected, name):
    plot.set_title(name)
    plot.plot(output, label='predicted')
    plot.plot(expected, label='true')
    plot.legend()
