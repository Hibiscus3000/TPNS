from metrics import *
from logging import *

def sort(normalized_attributes, normalized_targets, training_data_per,
         entropy_bottom_line):
    entropy = 0
    logger = getLogger(__name__)
    logger.info("finding good sample distribution, entropy bottom line: %f",
                entropy_bottom_line)
    while entropy < entropy_bottom_line:
        ids_shuffle = random.permutation(list(normalized_attributes.keys()))
        first_test_sample = int(training_data_per * len(normalized_attributes) \
                                / 100)
        learning_ids = ids_shuffle[0:first_test_sample]
        test_ids = ids_shuffle[first_test_sample:]

        attributes_entropy = get_samples_entropy(normalized_attributes,
                                                 learning_ids)
        targets_entropy = get_samples_entropy(normalized_targets, learning_ids)
        entropy = attributes_entropy + targets_entropy
        logger.debug("trying distributions with entropies: attributes - %f, "
                     "targets - %f", attributes_entropy, targets_entropy)

    logger.info("found good distribution with entropies: attributes - %f, "
                     "targets - %f", attributes_entropy, targets_entropy)
    return learning_ids, test_ids


def get_samples_entropy(all_samples, ids):
    tested_samples = [sample for sample_id, sample \
                      in all_samples.items() if
                      sample_id in ids]
    tested_attributes = transpose(tested_samples)
    entropy = 0
    for tested_attribute in tested_attributes:
        entropy += get_entropy(list(categories_to_sample_amount(split_intervals(
            tested_attribute)).values()))

    return entropy
