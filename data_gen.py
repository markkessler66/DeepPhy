import os
import sys
import random


def data_maker(num_data_sets, treename):
    myfile = open('mysh.sh','a')

    std_fs = [.2122, .2888, .2896, .2104]
    std_rs = [0.2173, 0.9798, 0.2575, 0.1038, 1, 0.2070]

    epsilon = .05
    
    newlines = []
    file_count = 0
    file_ext = '.tre'
    

    for dummy in range(num_data_sets):
        output_file = f'data/{treename}{file_count}.dat'

        
        new_rs = []

        
        for item in std_rs:
            rand_add = (random.random() - 0.5) * epsilon
            new_rs.append(rand_add + item)
        
        newlines.append(str_gen(new_rs, std_fs, f'{treename}{file_ext}', output_file, num = ' -n8000'))
        file_count += 1 

    
    myfile.writelines(newlines)
    



def str_gen(matrix, f_vals, treename, output_file, s_val = ' -s0.018', length = ' -l200', num = ' -n5000', model =' -mGTR'):
    mystr = 'source/seq-gen '
    matrix_params = ' -r'
    for item in matrix:
        matrix_params += f'{item},'
    matrix_params = matrix_params[0:-1]

    fmatrix_params = ' -f'
    for item in f_vals:
        fmatrix_params += f'{item},'
    fmatrix_params = fmatrix_params[0:-1]

    mystr = mystr + model + s_val + fmatrix_params + matrix_params + length + num +  f' <{treename}> {output_file}'
    return mystr + '\n'


data_maker(1, 'alpha')
data_maker(1, 'beta')
data_maker(1, 'charlie')
