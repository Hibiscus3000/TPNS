import os
import json
import logging
import logging.config
import pickle

from cell import *
import diagram_builder
from metrics import *
from reader.reader import *

def count_log_metrics(output, expected, name):
    cmae = mae(output, expected)
    cmse = mse(output, expected)
    crmse = rmse(output, expected)
    cr2 = r2(output, expected)
    cmape = mape(output, expected)

    logging.getLogger().info(name)
    logging.getLogger().info('MAE {:13.7f}'.format(cmae))
    logging.getLogger().info('MSE {:13.7f}'.format(cmse))
    logging.getLogger().info('RMSE {:13.7f}'.format(crmse))
    logging.getLogger().info('R2 {:13.7f}'.format(cr2))
    logging.getLogger().info('MAPE {:13.7f}'.format(cmape))

if '__main__' == __name__:
    with open(f'logging.json', 'r') as logging_config_file:
        logging_config = json.load(logging_config_file)
    logging.config.dictConfig(logging_config)

    if False == os.path.exists('pickle'):
        os.mkdir('pickle')

    with open('config.json','r') as config_file:
        config = json.load(config_file)

    window_size = config['window_size']

    reader = Reader(window_size)
    test_attributes, test_targets = reader.read_test()
    train_attributes, train_targets = reader.read_train()

    if config['clear_nn']:
        cell = get_cell_by_name(config['nn_type'], window_size)
    else:
        with open(config['pickle_file'], 'rb') as nn_file:
            cell = pickle.load(nn_file)

    epoch = config['epoch']

    for e in range(0, epoch):
        train_output = cell.train_all(config['learning_rate'],train_attributes, train_targets)
        test_output = cell.test_all(test_attributes,np.transpose(test_targets).shape)

        train_output_decoded = reader.decode(train_output)
        test_output_decoded = reader.decode(test_output)

        logging.getLogger().info(f'                 EPOCH {e + 1} / {epoch}')
        count_log_metrics(train_output_decoded, train_targets, 'TRAIN')
        count_log_metrics(test_output_decoded, test_targets, 'TEST')

    train_output = cell.test_all(train_attributes, np.transpose(train_targets).shape)
    test_output = cell.test_all(test_attributes, np.transpose(test_targets).shape)

    train_output_decoded = reader.decode(train_output)
    test_output_decoded = reader.decode(test_output)

    logging.getLogger().info('TOTAL')
    count_log_metrics(train_output_decoded, train_targets, 'TRAIN')
    count_log_metrics(test_output_decoded, test_targets, 'TEST')

    diagram_builder.show_prediction_results(train_output_decoded, reader.decode(train_targets),
                                            test_output_decoded, reader.decode(test_targets))
    
    if epoch:
        if 'y' == input('Do you want to save nn?[y]'):
            with open(config['pickle_file'], 'wb') as nn_file:
                pickle.dump(cell, nn_file)
