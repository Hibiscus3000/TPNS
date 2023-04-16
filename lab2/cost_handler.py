
class CostHandler:
    def __init__(self):
        self.learning_costs = {}
        self.testing_costs = {}

    def add_costs(self, is_learning, costs):
        supplement_costs = self.learning_costs if is_learning else self.testing_costs
        for cost in costs:
            for i in range(0, len(cost)):
                if cost[i] is not None:
                    if i not in supplement_costs:
                        supplement_costs[i] = [cost[i]]
                    else:
                        supplement_costs[i].append(cost[i])