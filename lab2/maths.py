from numpy import *

def divide_dicts(dict1, dict2):
    quotient = array(dict1.values()) / array(dict2.values())
    return get_dict(dict1, quotient)

def substract_dicts(dict1, dict2):
    quotient = array(dict1.values()) - array(dict2.values())
    return get_dict(dict1, quotient)

def get_dict(dict1, result):
    result_dict = {}
    i = 0
    for k, _ in dict1:
        result_dict[k] = result[i]
        i += 1