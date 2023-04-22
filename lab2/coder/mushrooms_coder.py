from logging import *

from coder.coder import Coder


class MushroomsCoder(Coder):

    def encode_attributes(self, samples):
        attributes = {}

        for sample in samples:
            for i in range(0, len(sample)):
                attribute = ord(sample[i]) - ord('a') + 1 if sample[i] is not None else 0
                if i not in attributes:
                    attributes[i] = [attribute]
                else:
                    attributes[i].append(attribute)
        getLogger(__name__).info("encoded %d attributes", len(attributes))
        return attributes

    def encode_targets(self, samples):
        return [[1 if 'e' == target[0] else 0 for target in samples]]

    def decode_attrs(self, attributes):
        return [self.decode_attribute(attribute) for attribute in attributes]

    def decode_attribute(self, attribute):
        return chr(attribute - 1 + ord('a')) if attribute != 0 else '?'

    def decode_targets(self, targets):
        return [self.decode_target(target) for target in targets]

    def decode_target(self, target):
        return 'e' if target > 0.5 else 'p'
