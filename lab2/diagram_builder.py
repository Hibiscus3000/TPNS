import matplotlib as mtl
from matplotlib import pyplot as plt


def show_cost_diagram(costs_learning, costs_testing, max_iterations):
    mtl.use('Qt5Agg')
    fig = plt.figure(figsize=(20, 12))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    fig.suptitle('Cost')
    columns = 2
    rows = len(costs_learning)
    for i in range(0, len(costs_learning)):
        plot = fig.add_subplot(100 * rows + 10 * columns + 2*i + 1)
        plot.title.set_text('learning ' + str(i))
        plt.xlabel('number of iterations')
        plt.ylabel('cost function')
        cost = costs_learning[i]
        number_of_costs = len(cost)
        first_iteration = max(0, number_of_costs - max_iterations)
        plt.plot(list(range(first_iteration, number_of_costs)),cost[-max_iterations:])

    for i in range(0, len(costs_testing)):
        plot = fig.add_subplot(100 * rows + 10 * columns + 2 * i + 2)
        plot.title.set_text('predicting ' + str(i))
        plt.xlabel('number of iterations')
        plt.ylabel('cost function')
        cost = costs_testing[i]
        number_of_costs = len(cost)
        first_iteration = max(0, number_of_costs - max_iterations)
        plt.plot(list(range(first_iteration, number_of_costs)), cost[-max_iterations:])

    plt.show()
