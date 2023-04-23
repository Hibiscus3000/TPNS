from numpy import *

# element of categories is the number of values in that category
def get_entropy(categories):
    entropy = 0
    samples_total = sum(categories)
    for category in categories:
        p = category / samples_total
        entropy -= p * math.log(p, 2)
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
        if dependent_category not in dependent_categories:
            dependent_categories[dependent_category] = 0
        else:
            dependent_categories[dependent_category] += 1

    samples_total = len(dependent_samples)
    conditional_entropy = 0
    on_which_depend_tot = len(on_which_depend)
    for num_of_dependent_in_category in dependent_categories.values():
        conditional_entropy -= num_of_dependent_in_category / samples_total *\
                               math.log(num_of_dependent_in_category / on_which_depend_tot,2)

    return conditional_entropy

def split_intervals(samples):
    samples_total = len(samples)
    intervals_total = 1 + int(math.log(samples_total, 2))
    interval_length = 1.0 / intervals_total
    intervals = {}
    for i in range(0, len(samples)):
        if samples[i] is not None:
            interval_id = int(samples[i] / interval_length)
            if interval_id not in intervals:
                intervals[interval_id] = [i]
            else:
                intervals[interval_id].append(i)
    return intervals


def samples_to_attributes(all_samples, ids):
    tested_samples = [sample for sample_id, sample \
                      in all_samples.items() if
                      sample_id in ids]
    return transpose(tested_samples)


def categories_to_samples(samples_to_categories):
    categories_to_samples = {}
    for sample_id, category in samples_to_categories.items():
        if category not in categories_to_samples:
            categories_to_samples[category] = [sample_id]
        else:
            categories_to_samples[category].append(sample_id)
    return categories_to_samples


def attribute_to_categories_to_samples(attribute):
    categories_to_samples = {}
    for sample_id in range(0, len(attribute)):
        if attribute[sample_id] not in categories_to_samples:
            categories_to_samples[attribute[sample_id]] = [sample_id]
        else:
            categories_to_samples[attribute[sample_id]].append(sample_id)
    return categories_to_samples


def samples_to_categories(categories_to_samples):
    samples_to_cat = {}
    for category, samples in categories_to_samples.items():
        for sample in samples:
            samples_to_cat[sample] = category
    return samples_to_cat


def categories_to_sample_amount(categories_to_samples):
    categories_to_sample_amount = {}
    for category, samples in categories_to_samples.items():
        categories_to_sample_amount[category] = len(samples)
    return categories_to_sample_amount

def get_info_gain(categories_dependent, categories_on_which_depend):
    entropy = get_entropy(list(categories_to_sample_amount(categories_dependent).values()))
    dependent_samples_to_categories = samples_to_categories(categories_dependent)
    conditional_normalized_entropy = 0
    for _, category_on_which_depend in categories_on_which_depend.items():
        conditional_normalized_entropy += get_normalized_conditional_entropy(
            dependent_samples_to_categories,
            category_on_which_depend)
    return entropy - conditional_normalized_entropy


def get_gain_ratio(categories_dependent, categories_on_which_depend):
    info_gain = get_info_gain(categories_dependent, categories_on_which_depend)
    intrinsic_info = get_entropy(list(categories_to_sample_amount(categories_on_which_depend).values()))
    return info_gain / intrinsic_info


def get_sample_mean(attribute):
    return average([at for at in attribute if at is not None])


def get_dispersion(attribute):
    return get_sample_mean((attribute - get_sample_mean(attribute)) ** 2)


def get_correlation(categories_dependent, categories_on_which_depend):
    covariance = 0
    categories_dependent_values = list(categories_dependent.values())
    dependent_sample_mean = get_sample_mean(categories_dependent_values)
    categories_on_which_depend_values = list(categories_on_which_depend.values())
    on_which_depend_sample_mean = get_sample_mean(categories_on_which_depend_values)
    for sample_id, category in categories_dependent.items():
        covariance += (category - dependent_sample_mean) \
                      * (categories_on_which_depend[sample_id] - on_which_depend_sample_mean)
    covariance /= len(categories_dependent.items())
    dependent_dispersion = get_dispersion(categories_dependent_values)
    on_which_depend_dispersion = get_dispersion(categories_on_which_depend_values)

    return covariance / sqrt(dependent_dispersion * on_which_depend_dispersion)
