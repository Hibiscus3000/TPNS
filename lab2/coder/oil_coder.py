from logging import *

import numpy as np

from coder.coder import Coder


class OilCoder(Coder):

    def encode(self, samples, withNones):
        attributes = {}

        for sample in samples:
            for i in range(0, len(sample)):
                if withNones | (sample[i] is not None):
                    attribute = float(sample[i]) if sample[
                                                        i] is not None else None
                    if i not in attributes:
                        attributes[i] = [attribute]
                    else:
                        attributes[i].append(attribute)
        getLogger(__name__).info("encoded %d attributes", len(attributes))
        return attributes

    def normalize_attribute(self, attribute):
        min = np.min(attribute)
        max = np.max(attribute)
        delta = max - min
        return min, delta, (np.array(attribute) - min) / delta

    def normalize(self, attributes):
        deltas = []
        mins = []
        normalized_attributes = []
        for attribute in attributes:
            min, delta, normalized_attribute = self.normalize_attribute(
                attribute)
            mins.append(min)
            deltas.append(delta)
            normalized_attributes.append(normalized_attribute.tolist())
        getLogger(__name__).info("normalized %d attributes", len(normalized_attributes))
        return deltas, mins, normalized_attributes

    def normalize_targets(self, targets):
        no_nones_targets = [[t for t in target if t is not None] for target in
                            targets]
        self.deltas, self.mins, normalized_targets = self.normalize(
            no_nones_targets)
        for j in range(0, len(targets)):
            for i in range(0, len(targets[j])):
                if targets[j][i] is None:
                    normalized_targets[j].insert(i, None)
        getLogger(__name__).info("normalized %d targets",
                                 len(normalized_targets))
        return normalized_targets

    def decode_targets(self, targets):
        return [targets[i] * self.deltas[i] + self.mins[i] if targets[i] is not None else None
                for i in range(0, len(self.mins))]
