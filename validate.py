#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:26:39 2020

@author: iikkateivas
"""
import argparse
import json
import matplotlib.pyplot as plt
import os

from data_utils import generate_data
from model import getModel

def validate(model, x_test, y_test, train_cfg):
    """
    Model validation function. Saves validation results to the train_cfg
    Arguments:
    x_test -- Test samples
    y_test -- Test lables
    train_cfg -- training config
    """
    
    # Add results into the train_cfg
    train_cfg['results'] = {
    'metric 1': 123,
    'metric 2': 456
    }
    
    # Save results as a new json file
    results_root = train_cfg['validation']['results_root']
    filename = os.path.join(results_root, 'results_' + train_cfg['test_name'] + '.json')
    with open(filename, 'w') as outfile:
        json.dump(train_cfg, outfile)
    
def main():
    """
    This is for separate validation run
    """   
    parser = argparse.ArgumentParser(description='Description')

    parser.add_argument('train_cfg', metavar='TRAIN',
                        help='path to train json')
    parser.add_argument('weights', metavar='TEST',
                        help='model weights')
    
    args = parser.parse_args()

    with open(args.train_cfg, 'r') as outfile:   
        train_cfg = json.load(outfile)
        
    _, _, x_test, _, _, y_test = generate_data(sets=args.datasets)
    
    model = getModel()
    model.load_weights(args.weights)
    
    validate(model, x_test, y_test, train_cfg)
    

if __name__ == '__main__':
    main()
