import json
import sys

from layer import *
from activation_function import *
from cnn import *
from nn_part import *

def get_activation_function(af_name):
    match af_name:
        case 'sigmoid':
            return Sigmoid()
        case _:
            sys.exit("Unknown activation function: {}".format(af_name))

def build_cnn():
    cnn = CNN()
    with open('cnn_config.json','r') as config_file:
        config = json.load(config_file)
    for layer in config:
        match layer['type']:
            case 'convolution':
                cnn.add_part(ConvolutionPart(ConvolutionLayer(image_depth=layer['depth'], filters=layer['filters'],size=layer['size'],
                                               p=layer['padding']),get_activation_function(layer['activation_function'])))
            case 'hidden':
                cnn.add_part(HiddenPart(HiddenLayer(number_of_neurons=layer['neurons'],
                                                    number_of_neurons_prev_layer= layer['previous_neurons']),
                                        get_activation_function(layer['activation_function'])))
            case 'output':
                cnn.add_part(OutputPart(OutputLayer(number_of_neurons=layer['neurons'],
                                                    number_of_neurons_prev_layer= layer['previous_neurons']),
                                        get_activation_function(layer['activation_function'])))
            case 'avg_pool':
                cnn.add_part(PoolingPart(AvgPoolLayer(image_depth=layer['depth'], prev_size = layer['prev_size'],
                                           size=layer['size'],stride=layer['stride'])))
            case _:
                sys.exit("unknown layer type: {}".format(layer['type']))
    return cnn