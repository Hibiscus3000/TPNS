from layer import *
from activation_function import *
from cnn import *
import json
import sys

def build_cnn():
    cnn = CNN()
    with open('config.json','r') as config_file:
        config = json.load(config_file)
    for cnn_part in config:
        match cnn_part['type']:
            case 'input':
                prev_size, prev_chnl = cnn_part['size'], cnn_part['channels']
                prev_neurons = prev_size * prev_chnl
            case 'convolution':
                cnn.add_layer(ConvolutionLayer(image_depth=prev_chnl, filters=cnn_part['filters'],height=cnn_part['size'],
                                               width=cnn_part['size'],p=cnn_part['padding']))
                prev_size, prev_chnl = cnn_part['size'], cnn_part['filters']
                prev_neurons = prev_size * prev_chnl
            case 'hidden':
                cnn.add_layer(HiddenLayer(number_of_neurons=cnn_part['neurons'], number_of_neurons_prev_layer= prev_neurons))
                prev_neurons = cnn_part['neurons']
            case 'output':
                cnn.add_layer(OutputLayer(number_of_neurons=cnn_part['neurons'], number_of_neurons_prev_layer= prev_neurons))
                prev_neurons = cnn_part['neurons']
            case 'avg_pool':
                cnn.add_layer(AvgPoolLayer(prev_chnl=prev_chnl, prev_height=prev_size, prev_width=prev_size,
                                           height=cnn_part['size'],width=cnn_part['size'],s1=cnn_part['stride'],s2=cnn_part['stride']))
            case 'sigmoid':
                cnn.add_activation_function(Sigmoid())
            case _:
                sys.exit("unknown nn part type: {}".format(cnn_part['type']))
    return cnn