from coder.coder import Coder
import numpy as np


class OilCoder(Coder):

    def encode_attributes(self, samples):
        attributes = {}

        for sample in samples:
            for i in range(0, len(sample)):
                if (None != sample[i]):
                    if i not in attributes:
                        attributes[i] = [float(sample[i])]
                    else:
                        attributes[i].append(float(sample[i]))
        return attributes

    def encode_targets(self, samples):
        targets = {}

        for sample in samples:
            for i in range(0, len(sample)):
                if (None != sample[i]):
                    if i not in targets:
                        targets[i] = [float(sample[i])]
                    else:
                        targets[i].append(float(sample[i]))
        return targets

    def normalize(self, attributes):
        deltas = []
        mins = []
        attr_arrs = np.array(attributes)
        for i in range(0, len(attributes)):
            print(attr_arrs)
            min = np.min(attr_arrs[i])
            max = np.max(attr_arrs[i])
            mins.append(min)
            deltas.append(max - min)
        return deltas, mins, (np.transpose(attr_arrs) - mins) / deltas

    def normalize_targets(self, targets):
        deltas, mins, normalized_targets = self.normalize([t for t in
                                                           targets if t
                                                           is not None])
        for i in range(0, len(targets)):
            if targets[i] is None:
                normalized_targets.insert(i, None)

        return deltas, mins, normalized_targets

    def decode(self, deltas, targets):
        return np.transpose(targets * deltas)
