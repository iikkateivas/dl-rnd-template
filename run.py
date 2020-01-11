# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:26:39 2020

@author: iikkateivas
"""

import numpy as np
import os
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop, SGD
import argparse
import json

from data_utils import generate_data, pre_process_data, generate_batches
from model import getModel
from validate import validate


def main():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('train_cfg', metavar='TRAIN',
                        help='path to train json')    
    args = parser.parse_args()
    
    with open(args.train_cfg, 'r') as outfile:   
        train_cfg = json.load(outfile)
    
    # Data args
    args.test_name = train_cfg['test_name']
    args.datasets = train_cfg['data']['datasets']
    args.pre_norm = str_to_bool(train_cfg['data']['pre_norm'])
    
    # Training args
    args.type = train_cfg['training']['optimizer']['type']
    args.lr = train_cfg['training']['optimizer']['lr']
    args.decay = train_cfg['training']['optimizer']['decay']
    args.momentum = train_cfg['training']['optimizer']['momentum']   
    args.pre_trained = str_to_bool(train_cfg['training']['pre_trained'])
    args.weight_path = train_cfg['training']['weight_path']
    args.loss = train_cfg['training']['loss']
    args.metrics = train_cfg['training']['training_metrics']
    args.epochs = train_cfg['training']['epochs']
    args.batch_size = train_cfg['training']['batch_size']
    args.val_data = str_to_bool(train_cfg['training']['val_data'])
    
    args.save_weights_only = str_to_bool(train_cfg['training']['checkpoint']['save_weights_only'])
    args.save_best_only = str_to_bool(train_cfg['training']['checkpoint']['save_best_only'])
    args.period = train_cfg['training']['checkpoint']['period'] 

    # Validation args
    args.results_root = train_cfg['validation']['results_root']
    '''Generate data'''
    x_train, x_val, x_test, y_train, y_val, y_test = generate_data(sets=args.datasets)
    
    '''pre process data'''
    pre_process_data(x_train, x_val, x_test, y_train, y_val, y_test, str_to_bool(args.pre_norm))
    
    '''Get model'''
    model = getModel()
    print(model.summary())
    
    if args.type == 'sgd':
        opt = SGD(lr=args.lr, decay=args.decay, momentum=args.momentum, nesterov=True)
    elif args.type == 'adam':
        opt = Adam(lr=args.lr)
    elif args.type == 'rmsprop':
        opt = RMSprop(lr=args.lr, decay=args.decay)
    else:
        opt = SGD(lr=args.lr, decay=args.decay, momentum=args.momentum, nesterov=True)
    
    
    model.compile(optimizer=opt,loss=args.loss, metrics=args.metrics)
    
    if str_to_bool(args.pre_trained):
        model.load_weights(args.weight_path)
    
    weight_dir =  args.test_name
    weight_root = os.path.join('weights', weight_dir)
    
    # Folder for intermediate weights
    if not os.path.exists(weight_root):
        os.makedirs(weight_root)
    
    
    epochs = args.epochs
    steps = int(len(x_train)/args.batch_size)
    validation_steps = int(len(x_val)/args.batch_size)
    
    if str_to_bool(args.val_data):
        filepath = os.path.join(weight_root, 'weights-{epoch:03d}-{val_loss:.4f}.h5')
        checkpoint = ModelCheckpoint(filepath, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_weights_only=args.save_weights_only, 
                                     save_best_only=args.save_best_only,  
                                     mode='auto', 
                                     period=args.period)
        
        model.fit_generator(generate_batches(x_train, y_train, args.batch_size, str_to_bool(args.pre_norm)),
                        validation_data=generate_batches(x_val, y_val, args.batch_size, str_to_bool(args.pre_norm)),
                        validation_steps=validation_steps,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        callbacks=[checkpoint])
    else:
        filepath = os.path.join(weight_root, 'weights-{epoch:03d}-{loss:.4f}.h5')
        checkpoint = ModelCheckpoint(filepath, 
                                     monitor='loss', 
                                     verbose=1, 
                                     save_weights_only=args.save_weights_only, 
                                     save_best_only=args.save_best_only,  
                                     mode='auto', 
                                     period=args.period)
        
        model.fit_generator(generate_batches(x_train, y_train, str_to_bool(args.pre_norm)),
                        steps_per_epoch=steps,
                        epochs=epochs,
                        callbacks=[checkpoint])
    
    '''Validation'''
    validate(model, x_test, y_test, train_cfg)

def str_to_bool(s):
    if s == 'True':
         return True
    else:
         return False

if __name__ == '__main__':
    main()