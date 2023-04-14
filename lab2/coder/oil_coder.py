from coder.coder import Coder
import numpy as np


class OilCoder(Coder):

    def encode(self, str_samples, withNones):
        samples = {}

        for i, str_attributes in str_samples:
            for str_attribute in str_attributes:
                if (None != str_attribute | withNones):
                    attribute = float(str_attribute) if str_attribute is not \
                                                      None else None
                    if i not in samples:
                        samples[i] = [float(attribute)]
                    else:
                        samples[i].append(float(attribute))
        return samples

    def get_mins_deltas(self,samples):
        deltas = {}
        mins = {}
        for sample_id, sample in samples:
            min = np.min(sample)
            max = np.max(sample)
            mins[sample]
            deltas.append(max - min)

        return mins, deltas

    def normalize_attributes(self, attributes):
        mins, deltas = self.get_mins_deltas(attributes)
        normalized_attrs = (attr_list - np.transpose(np.atleast_2d(mins))) / \
        np.transpose(np.atleast_2d(deltas))
        return deltas, mins,

    def normalize_targets(self, targets):
        no_nones_targets = {sample_id: [value for value in target_values if
                                        value is not None] for
                            sample_id, target_values in targets.items()}
        mins, deltas = self.get_mins_deltas(no_nones_targets)

        normalized_targets = {}
        for sample_id, sample in no_nones_targets
            targets
        for i in range(0, len(targets)):
            if targets[i] is None:
                normalized_targets.insert(i, None)

        return deltas, mins, normalized_targets

    def get_normalized_samples(self, normalized_attributes,
                               normalized_targets):
        normalized_samples = {}
        for sample_id, sample in normalized_attributes:
            normalized_samples[sample_id] = np.concatenate(sample,
                                                       normalized_targets[
                                                           sample_id])

        return normalized_samples

    def decode(self, deltas, targets):
        return np.transpose(targets * deltas)
