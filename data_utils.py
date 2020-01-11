# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:26:39 2020

@author: iikkateivas
"""

import numpy as np
from sklearn.utils import shuffle

def generate_data(sets = 'ab'):     
    
    x_train = None
    x_val = None
    x_test = None
    y_train = None
    y_val = None
    y_test = None

    return x_train, x_val, x_test, y_train, y_val, y_test

def pre_process_data(x_train, x_val, x_test, y_train, y_val, y_test, seed = 1337, norm = True):
    x_train = np.asarray(x_train)
    x_val = np.asarray(x_val)
    x_test = np.asarray(x_test)

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    
    print('Shuffling training data...')
    x_train, y_train = shuffle(x_train, y_train, random_state = seed)
    
    if norm:
        print('Normalizing data...')
        x_train = np.divide(x_train, 255, dtype=np.float32)
        x_val = np.divide(x_val, 255, dtype=np.float32)
        x_test = np.divide(x_test, 255, dtype=np.float32)
        
def generate_batches(x, y, batch_size, norm = True):
    while 1:
        # Generate batches here
        x_batch = None
        y_batch = None
        if not norm:
            x_batch = np.divide(x_batch, 255, dtype=np.float32)
            
        yield (x_batch, y_batch)
    