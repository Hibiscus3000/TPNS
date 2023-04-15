import numpy as np

def format_matrix_array(matrix_array):
    matrix_array_str = ''
    for i in range(0,len(matrix_array)):
        matrix_array_str += "\t" + str(i) + "\n" + format_matrix(np.atleast_2d(matrix_array[i]))
        if i != len(matrix_array) - 1:
            matrix_array_str += '\n'

    return matrix_array_str

def format_matrix(matrix):
    matrix_str = ''
    for l in range(0,len(matrix)):
        line_str = '|'
        for e in matrix[l]:
            line_str += f' {e:7.3f}'
        matrix_str += line_str + ' |'
        if l != len(matrix) - 1:
            matrix_str += '\n'
    return matrix_str
