from math import *

# element of categories is the number of values in that category
def get_entropy(categories, samples_total):
    entropy = 0
    for category in categories:
        p = category / samples_total
        entropy += p * log(p, 2)
    return entropy

# in dependent_samples key is the serial number of the sample
# value is the serial number of category
# in on_which_depend is a list of serial number of samples which are included in that
# category
def get_normalized_conditional_entropy(dependent_samples,
                                       on_which_depend):
    dependent_categories = {}
    for sample_ind in on_which_depend:
        dependent_category = dependent_samples[sample_ind]
        if dependent_category in dependent_categories:
            dependent_categories[dependent_category] = 0
        else:
            dependent_categories[dependent_category] += 1

    samples_total = len(dependent_samples)
    conditional_entropy = 0
    on_which_depend_tot = len(on_which_depend)
    for num_of_dependent_in_category in dependent_categories.values():
        conditional_entropy -= num_of_dependent_in_category / samples_total *\
                               log(num_of_dependent_in_category / on_which_depend_tot,2)

    return conditional_entropy


# samples is the dictionary, there key is the index of a sample and value is
# the attribute value of that sample
# normalized data expected
def split_intervals(samples):
    samples_total = len(samples)
    intervals_total = 1 + int(log(samples_total, 2))
    interval_length = 1.0 / intervals_total
    intervals = {}
    for index, sample in samples:
        if sample is not None:
            interval_id = int(sample / interval_length)
            if interval_id not in intervals:
                intervals[interval_id] = [sample]
            else:
                intervals[interval_id].append(index)
    return intervals

def samples_to_categories(categories_to_samples):
    samples_to_cat = {}
    for category, samples in categories_to_samples:
        for sample in samples:
            samples_to_cat[sample] = category
    return samples_to_cat

def categories_to_sample_amount(categories_to_samples):
    categories_to_sample_amount = {}
    for category, samples in categories_to_samples:
        categories_to_samples[category] = len(samples)
    return categories_to_sample_amount


def get_info_gain(categories_dependent, categories_on_which_depend):
    entropy = get_entropy(categories_to_sample_amount(categories_dependent))
    dependent_samples_to_categories = samples_to_categories(categories_dependent)
    conditional_normalized_entropy = 0
    for _, category_on_which_depend in categories_on_which_depend:
        conditional_normalized_entropy += get_normalized_conditional_entropy(
            dependent_samples_to_categories,
            category_on_which_depend)
    return entropy - conditional_normalized_entropy

def get_gain_ratio(categories_dependent, categories_on_which_depend):
    info_gain = get_info_gain(categories_dependent, categories_on_which_depend)
    intrinsic_info = get_entropy(categories_to_sample_amount(categories_on_which_depend))
    return info_gain / intrinsic_info
