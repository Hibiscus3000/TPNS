from coder.coder import Coder
import numpy as np


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
        return deltas, mins, normalized_attributes

    def normalize_targets(self, targets):
        no_nones_targets = [[t for t in target if t is not None] for target in
                            targets]
        deltas, mins, normalized_targets = self.normalize(
            no_nones_targets)
        for j in range(0, len(targets)):
            for i in range(0, len(targets[j])):
                if targets[j][i] is None:
                    normalized_targets[j].insert(i, None)

        return deltas, mins, normalized_targets

    def get_normalized_samples(self, sample_ids, attributes, targets):
        attrs_t = np.transpose(attributes)
        targ_t = np.transpose(targets)
        return {sample_ids[i]: attrs_t[i] for i in range(0, len(sample_ids))}, {
            sample_ids[i]: targ_t[i] for i
            in range(0, len(sample_ids))}

    def decode(self, deltas, targets):
        return np.transpose(targets * deltas)
