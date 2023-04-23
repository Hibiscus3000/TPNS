import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


def show_metrics(learning, predicting):
    for i in range(0, len(learning)):
        fig, axs = plt.subplots(2, len(learning[0]) - ('confusion matrix' in learning[0]),
                                figsize=(20, 12), constrained_layout=True)
        fig.suptitle(f'Metrics {i}')
        add_metrics_one_selection(axs[0], learning[i])
        add_metrics_one_selection(axs[1], predicting[i])
    plt.show()


def add_metrics_one_selection(axes, selection):
    i = 0
    for metric_name, metric in selection.items():
        if 'confusion matrix' == metric_name:
            continue
        axes[i].set_title(metric_name)
        axes[i].set_xlabel('epoch')
        axes[i].plot(metric)
        i += 1


def show_roc(tpr_learning, fpr_learning, tpr_predicting, fpr_predicting):
    fig, axs = plt.subplots(1, 2, figsize=(20, 12), constrained_layout=True)
    add_roc_one_selection(axs[0], tpr_learning, fpr_learning)
    add_roc_one_selection(axs[1], tpr_predicting, fpr_predicting)
    fig.suptitle('ROC')
    plt.show()


def add_roc_one_selection(axes, tpr, fpr):
    axes.plot(fpr, tpr)
    axes.set_xlabel('FPR')
    axes.set_ylabel('TPR')


def show_prediction_results(learning_ids, results_learning, expected_learning,
                            predicting_ids, results_predicting, expected_predicting):
    fig, axs = plt.subplots(2, len(results_learning), figsize=(20, 12), constrained_layout=True)
    add_pre_res_one_selection(axs[0], learning_ids,
                              results_learning, expected_learning)
    add_pre_res_one_selection(axs[1], predicting_ids,
                              results_predicting, expected_predicting)
    plt.legend()
    fig.suptitle('Predicting results')
    plt.show()


def add_pre_res_one_selection(axes, ids, results, expected):
    for i in range(0, len(results)):
        axes[i].set_title(i)
        axes[i].plot(ids, results[i], marker='o', label='predicted')
        axes[i].plot(ids, expected[i], marker='o', label='true')
        axes[i].set_xlabel('ids')
