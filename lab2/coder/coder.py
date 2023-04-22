import abc

import numpy as np


class Coder(abc.ABC):

    @abc.abstractmethod
    def decode_targets(self, results):
        """decode perceptron results"""

    def get_samples(self, sample_ids, attributes, targets):
        attrs_t = np.transpose(attributes)
        targ_t = np.transpose(targets)
        return {sample_ids[i]: attrs_t[i] for i in range(0, len(sample_ids))}, {
            sample_ids[i]: targ_t[i] for i
            in range(0, len(sample_ids))}
