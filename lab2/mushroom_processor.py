from main import *
from metrics import *


def clear_attributes(attribute_samples, target_samples, ids, critical_correlation):
    attributes = samples_to_attributes(attribute_samples, ids)
    targets = samples_to_attributes(target_samples, ids)
    categories_to_samples_attrs = []
    categories_to_samples_tars = []
    for i in range(0, len(attributes)):
        categories_to_samples = attribute_to_categories_to_samples(attributes[i])
        if 0 != get_entropy(list(categories_to_sample_amount(categories_to_samples).values())):
            categories_to_samples_attrs.append(categories_to_samples)
        else:
            getLogger(__name__).info(f'removed attribute № {i} due to zero entropy')
    for target in targets:
        categories_to_samples_tars.append(attribute_to_categories_to_samples(target))

    gain_ratios = [[get_gain_ratio(categories_to_samples_tar, categories_to_sample_amount_attr)
                    for categories_to_sample_amount_attr in categories_to_samples_attrs]
                   for categories_to_samples_tar in categories_to_samples_tars]
    samples_to_categories_attr = [samples_to_categories(categories_to_samples)
                                  for categories_to_samples in categories_to_samples_attrs]
    samples_to_categories_tar = [samples_to_categories(categories_to_samples)
                                 for categories_to_samples in categories_to_samples_tars]
    correlations_attrs = [[get_correlation(samples_to_categories_attr[j], samples_to_categories_attr[i])
                           for j in range(0, i)]
                          for i in range(0, len(samples_to_categories_attr))]
    correlation_attrs_targets = [[get_correlation(samples_to_categories_attr[j], samples_to_categories_tar[i])
                                  for j in range(0, len(samples_to_categories_attr))]
                                 for i in range(0, len(samples_to_categories_tar))]

    to_remove = {}
    for i in range(0, len(samples_to_categories_attr)):
        for j in range(0, i):
            if abs(correlations_attrs[i][j]) > critical_correlation:
                number_of_defeats = 0
                for t in range(0, len(samples_to_categories_tar)):
                    if correlation_attrs_targets[t][i] < correlation_attrs_targets[t][j]:
                        number_of_defeats += 1
                    if gain_ratios[t][i] < gain_ratios[t][j]:
                        number_of_defeats += 1
                if number_of_defeats > len(correlation_attrs_targets):
                    to_remove[i] = True
                    to_remove[j] = False
                else:
                    to_remove[j] = True
                    to_remove[i] = False

    for attribute_id, need_to_remove in to_remove.items():
        if need_to_remove:
            del samples_to_categories_attr[attribute_id]
            getLogger(__name__).info("removed attribute № {} due to high correlation with other attributes"
                                     .format(attribute_id))
    return samples_to_categories_attr, samples_to_categories_tar


def clear_samples(samples_to_categories_attr, emission_threshold):
    emissions = {}
    for attribute in samples_to_categories_attr:
        attribute_list = list(attribute.values())
        sample_mean = get_sample_mean(attribute_list)
        dispersion_sqrt = math.sqrt(get_dispersion(attribute_list))
        for sample_id, value in attribute.items():
            if abs(value - sample_mean) > emission_threshold * dispersion_sqrt:
                emissions[sample_id] = True

    getLogger(__name__).info("-----------------------------------------------------------------")
    to_remove_samples = []
    for sample_id, number_of_emissions in emissions.items():
        if emissions[sample_id]:
            getLogger(__name__).info(f"removing sample № {sample_id} due to emission")
            to_remove_samples.append(sample_id)
    getLogger(__name__).info("-----------------------------------------------------------------")
    getLogger(__name__).info("samples removed total: {}".format(len(to_remove_samples)))
    return to_remove_samples


def write_to_file(filename, samples_to_categories_attr, samples_to_categories_tars, ids, coder, to_remove_ids):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        for sample_id in ids:
            if sample_id in to_remove_ids:
                continue
            row = []
            for i in range(0, len(samples_to_categories_tars)):
                row.append(coder.decode_target(samples_to_categories_tars[i][sample_id]))
            for i in range(0, len(samples_to_categories_attr)):
                row.append(coder.decode_attribute(samples_to_categories_attr[i][sample_id]))
            writer.writerow(row)


if __name__ == '__main__':
    config = read_config('mushrooms')
    is_oil, str_attrs, str_targets = read_data(config['reader'])
    if False == is_oil:
        coder, _, _, attributes, targets = code_data(str_attrs, str_targets, is_oil)
    processor_config = config['processor']
    samples_to_categories_attr, samples_to_categories_tars = \
        clear_attributes(attributes, targets, targets.keys(), processor_config['critical_correlation'])

    to_remove_ids = clear_samples(samples_to_categories_attr, processor_config['emission_threshold'])

    write_to_file(config['processor']['output_data_file'], samples_to_categories_attr,
                  samples_to_categories_tars, attributes.keys(), coder, to_remove_ids)
