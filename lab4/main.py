import os
import json
import logging
import logging.config

import pickle

from cnn_builder import *
from cnn import *
from diagram_builder import *
from reader import *
from metrics import *


def log_metric(metric_train, metric_test, metric_name):
    logging.getLogger(__name__).info(f'{metric_name}')
    logging.getLogger(__name__).info('TRAIN')
    for i in range(0, number_of_classes):
        logging.getLogger(__name__).info(f'{i}: {metric_train[i]}')
    logging.getLogger(__name__).info('TEST')
    for i in range(0, number_of_classes):
        logging.getLogger(__name__).info(f'{i}: {metric_test[i]}')


if '__main__' == __name__:
    if False == os.path.exists('pickle'):
        os.mkdir('pickle')

    with open(f'logging.json', 'r') as logging_config_file:
        logging_config = json.load(logging_config_file)
    logging.config.dictConfig(logging_config)

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    if config['nn']['clear']:
        cnn = build_cnn()
    else:
        with open(config['nn']['file'], 'rb') as nn_file:
            cnn = pickle.load(nn_file)

    if config['reader']['clear']:
        reader = Reader(config['reader']['train'], config['reader']['test'])
        with open(config['reader']['file'], 'wb') as reader_file:
            pickle.dump(reader, reader_file)
    else:
        with open(config['reader']['file'], 'rb') as reader_file:
            reader = pickle.load(reader_file)

    train = reader.get_train()
    test = reader.get_test()

    prev_train_accuracy = 0
    prev_test_accuracy = 0
    for i in range(0, config['nn']['epoch']):
        train_outputs = cnn.train(
            config['nn']['learning rate'], train[0], train[1])
        test_outputs = cnn.test(test[0])

        train_out_dec = reader.decode_all(train_outputs)
        train_exp_dec = reader.decode_all(train[1])
        logging.getLogger().debug('TRAIN')
        for j in range(0, len(train_out_dec)):
            output = ["{:.2f}".format(train_output) for train_output in train_outputs[j]]
            logging.getLogger().debug(f'{train_exp_dec[j]} - {output}')

        test_out_dec = reader.decode_all(test_outputs)
        test_exp_dec = reader.decode_all(test[1])
        logging.getLogger().debug('TEST')
        for j in range(0, len(test_out_dec)):
            output = ["{:.2f}".format(test_output) for test_output in test_outputs[j]]
            logging.getLogger().debug(f'{test_exp_dec[j]} - {output}')

        train_accuracy = accuracy(train_out_dec, train_exp_dec)
        test_accuracy = accuracy(test_out_dec, test_exp_dec)
        logging.getLogger(__name__).info(
            f'EPOCH {i} ACCURACY --- TRAIN: {train_accuracy} TEST: {test_accuracy}')
        #if (prev_test_accuracy > test_accuracy) | (prev_train_accuracy > train_accuracy):
        #    if 'y' != input('Do you want to continue?[y]'):
        #        cnn.rollback_prev_training(config['nn']['learning rate'])
        #        break
        prev_test_accuracy = test_accuracy
        prev_train_accuracy = train_accuracy

        with open(config['nn']['backup'], 'wb') as backup_file:
                pickle.dump(cnn, backup_file)

    train_outputs = cnn.test(train[0])
    test_outputs = cnn.test(test[0])
    logging.getLogger(__name__).info('METRICS')
    decoded_output_train = reader.decode_all(train_outputs)
    decoded_output_test = reader.decode_all(test_outputs)
    decoded_expected_train = reader.decode_all(train[1])
    decoded_expected_test = reader.decode_all(test[1])

    train_accuracy = accuracy(decoded_output_train, decoded_expected_train)
    test_accuracy = accuracy(decoded_output_test, decoded_expected_test)
    logging.getLogger(__name__).info('ACCURACY')
    logging.getLogger(__name__).info(f'test: {test_accuracy}, train: {train_accuracy}')

    cms_train = confusion_matrix_all(
        decoded_output_train, decoded_expected_train)
    cms_test = confusion_matrix_all(decoded_output_test, decoded_expected_test)
    log_metric(cms_train, cms_test, 'CONFUSION MATRIX [TP, TN, FN, FP]')

    recalls_train = metric_all(cms_train, recall)
    recalls_test = metric_all(cms_test, recall)
    log_metric(recalls_train, recalls_test, 'RECALL')

    precisions_train = metric_all(cms_train, precision)
    precisions_test = metric_all(cms_test, precision)
    log_metric(precisions_train, precisions_test, 'PRECISION')

    f1_train = metric_all(cms_train, f1)
    f1_test = metric_all(cms_test, f1)
    log_metric(f1_train, f1_test, 'F1')

    rocs_train = roc_all(np.array(train_outputs), decoded_expected_train)
    rocs_test = roc_all(np.array(test_outputs), decoded_expected_test)
    show_roc(rocs_train, rocs_test)

    if config['nn']['epoch']:
        if 'y' == input('Do you want to save nn?[y]'):
            with open(config['nn']['file'], 'wb') as nn_file:
                pickle.dump(cnn, nn_file)
