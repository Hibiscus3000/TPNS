import json
import logging.config

import pickle
from coder.mushrooms_coder import MushroomsCoder
from coder.oil_coder import OilCoder
from csv_reader import *
from meter import *
from perceptron import *
from sorter import *


def read_config(name):
    np.set_printoptions(precision=3)
    with open('config/logging.json', 'r') as logging_config_file:
        logging_config = json.load(logging_config_file)
    with open(f'config/{name}_config.json', 'r') as config_file:
        config = json.load(config_file)
    logging.config.dictConfig(logging_config)
    getLogger().handlers[1].doRollover()
    return config


def read_data(reader_config):
    data_file_name = reader_config['data_file']
    is_oil = data_file_name.startswith('data/oil')
    reader = CsvReader(data_file_name)
    str_attr = reader.get_values(reader_config['start_row'],
                                 reader_config["end_row"],
                                 reader_config['start_attributes'],
                                 reader_config['end_attributes'])
    str_targets = reader.get_values(reader_config['start_row'],
                                    reader_config["end_row"],
                                    reader_config['start_targets'],
                                    reader_config['end_targets'])
    str_attr = {sample_id: attr_sample for sample_id, attr_sample in
                str_attr.items() if sample_id in str_targets}
    return is_oil, str_attr, str_targets


def code_data(str_attr, str_targets, is_oil):
    coder = OilCoder() if is_oil else MushroomsCoder()
    if is_oil:
        attributes = coder.encode(str_attr.values(), withNones=False)
        targets = coder.encode(str_targets.values(), withNones=True)
        _, _, norm_attr = coder.normalize(list(
            attributes.values()))
        norm_targets = coder.normalize_targets(list(targets.values()))
        normalized_attrs, normalized_targets = coder.get_samples(list(
            str_targets.keys()), norm_attr, norm_targets)
        return coder, len(norm_attr), len(norm_targets), normalized_attrs, normalized_targets
    else:
        attributes = coder.encode_attributes(str_attr.values())
        targets = coder.encode_targets(str_targets.values())
        attrs, targs = coder.get_samples(list(str_targets.keys()), list(attributes.values()), targets)
        return coder, len(attributes), len(targets), attrs, targs


def get_samples(sorter_config, normalized_attrs, normalized_targets):
    if sorter_config['clear_samples']:
        learning_ids, test_ids = sort(normalized_attrs, normalized_targets,
                                      sorter_config['training_data_per'],
                                      sorter_config['entropy_bottom_line'])
        with open(sorter_config['learning_file'], 'wb') as learning_file:
            pickle.dump(learning_ids, learning_file)
        with open(sorter_config['test_file'], 'wb') as test_file:
            pickle.dump(test_ids, test_file)
    else:
        getLogger(__name__).debug("obtaining previous learning and testing ids...")
        with open(sorter_config['learning_file'], 'rb') as learning_file:
            learning_ids = pickle.load(learning_file)
        with open(sorter_config['test_file'], 'rb') as test_file:
            test_ids = pickle.load(test_file)
    getLogger(__name__).debug("learning ids: [{}]".format(' '.join(map(str, learning_ids))))
    getLogger(__name__).debug("test ids: [{}]".format(' '.join(map(str, test_ids))))
    return learning_ids, test_ids


def get_perceptron(perceptron_config, number_of_input_neurons, number_of_output_neurons, meter, coder):
    if perceptron_config['clear_weights']:
        neurons = perceptron_config['neurons']
        neurons.insert(0, number_of_input_neurons)  # insert input layer
        neurons.append(number_of_output_neurons)  # appending output layer
        perceptron = Perceptron(neurons, meter, coder, perceptron_config['need_to_decode'])
    else:
        getLogger(__name__).debug("obtaining previous perceptron weight and biases")
        with open(perceptron_config['perceptron_file'], 'rb') as perceptron_file:
            perceptron = pickle.load(perceptron_file)
        perceptron.log_weights_biases()

    return perceptron


def get_samples_by_ids(samples, ids):
    return {id: samples[id] for id in ids}


def get_learning_rates(learning_rates):
    return {int(epoch): learning_rate for epoch, learning_rate in learning_rates.items()}


def get_meter(name, number_of_targets):
    return QualificationMeter(number_of_targets) if 'mushrooms' == name \
        else RegressionMeter(number_of_targets)


if __name__ == '__main__':
    config = read_config(sys.argv[1])
    is_oil, str_attr, str_targets = read_data(config['reader'])
    coder, number_of_input_neurons, number_of_output_neurons, normalized_attrs, normalized_targets = \
        code_data(str_attr, str_targets, is_oil)
    learning_ids, test_ids = get_samples(config['sorter'], normalized_attrs, normalized_targets)
    learning_attrs = get_samples_by_ids(normalized_attrs, learning_ids)
    learning_targets = get_samples_by_ids(normalized_targets, learning_ids)
    test_attrs = get_samples_by_ids(normalized_attrs, test_ids)
    test_targets = get_samples_by_ids(normalized_targets, test_ids)
    meter = get_meter(sys.argv[1], number_of_output_neurons)
    perceptron_config = config['perceptron']
    perceptron = get_perceptron(perceptron_config, number_of_input_neurons, number_of_output_neurons,
                                meter, coder)
    if perceptron_config['only_predict']:
        attrs = {**learning_attrs, **test_attrs}
        targets = {**learning_targets, **test_targets}
        outputs = perceptron.predict(attrs.items())
    else:
        perceptron.learn_and_predict(perceptron_config['epoch'],
                                     learning_attrs,
                                     learning_targets,
                                     test_attrs,
                                     test_targets,
                                     get_learning_rates(perceptron_config['learning_rates']),
                                     perceptron_config['iterations_on_last_epoch'])
        if 'y' == input('Do you want to save weights, biases and metrics?[y]'):
            getLogger(__name__).debug('saving weights, biases and metrics to file')
            with open(perceptron_config['perceptron_file'], 'wb') as perceptron_file:
                pickle.dump(perceptron, perceptron_file)
            with open(perceptron_config['meter_file'], 'wb') as meter_file:
                pickle.dump(meter, meter_file)
