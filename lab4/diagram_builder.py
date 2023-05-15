import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


def show_roc(rocs_training, rocs_testing):
    fig, axs = plt.subplots(1, 2, figsize=(20, 12), constrained_layout=True)
    add_roc_one_selection(axs[0], rocs_training, 'training')
    add_roc_one_selection(axs[1], rocs_testing, 'testing')
    fig.suptitle('ROC')
    plt.show()


def add_roc_one_selection(axes, rocs, name):
    for i in range(0, len(rocs)):
        axes.plot(rocs[i][1], rocs[i][0], label=str(i))
    axes.title.set_text('ROC')
    axes.set_xlabel('FPR')
    axes.set_ylabel('TPR')
    axes.legend()
