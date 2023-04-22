import matplotlib as mtl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

def show_component_cost_diagram(costs_learning, costs_testing, max_iterations):
    mtl.use('Qt5Agg')
    fig = plt.figure(figsize=(20, 12))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    columns = 2
    rows = len(costs_learning)
    for i in range(0, len(costs_learning)):
        add_one_plot(fig, 100 * rows + 10 * columns + 2 * i + 1,
                     costs_learning[i], max_iterations,'learning ' + str(i),'number of iterations')

    for i in range(0, len(costs_testing)):
        add_one_plot(fig, 100 * rows + 10 * columns + 2 * i + 2,
                     costs_testing[i], max_iterations, 'predicting ' + str(i), 'number of iterations')

    plt.show()

def add_one_plot(fig, index, cost, max_values, title, xlabel):
    plot = fig.add_subplot(index)
    plot.title.set_text(title)
    plt.xlabel(xlabel)
    plt.ylabel('cost function')
    number_of_costs = len(cost)
    first_iteration = max(0, number_of_costs - max_values) if 0 != max_values else 0
    plot.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(list(range(first_iteration, number_of_costs)), cost[-max_values:])

def show_cost_diagram(learning_cost, testing_cost, max_epoch):
    mtl.use('Qt5Agg')
    fig = plt.figure(figsize=(20, 12))
    fig.subplots_adjust(hspace=0.4)

    #learning cost
    add_one_plot(fig, 211, learning_cost, max_epoch, 'learning', 'epoch')

    #predicting cost
    add_one_plot(fig, 212, testing_cost, max_epoch, 'predicting', 'epoch')

    plt.show()