import setGPU
import os
import argparse
import h5py
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import keras_tuner
from keras_tuner import HyperParameters
from tensorboard import program
from tensorflow.keras.callbacks import (EarlyStopping,ReduceLROnPlateau,TensorBoard)
import utils.data_processing as data_proc
from nn.models import GraphAttentionHyperModel
import nn.losses as nn_losses

fixed_seed = 2021
tf.keras.utils.set_random_seed(fixed_seed)


def main_optimize_l1_teacher(data_file='',variable='',log_features=[''], loss_function='',metric_thresholds='',
                            test_split=0.2,batch_size=1024,max_epochs=1,hyperband_factor=3,
                            output_dir=''):
    """
    Performs optimization of the teacher model
    Arguments:
        data_file: str, path to the input file 
        variable: str, variable for which we train : MET or HT 
        log_features: list of str, which feature scale to be log
        loss_function: str or loss function object
        metric_thresholds: list of floats, thresholds to be monitored in the MseThesholdMetric metric
        test_split: float, fraction to use for testing during training 
        batch_size: int, batch size
        max_epochs: int, max epochs
        hyperband_factor: int, hyperband_factor
        output_dir: str, output directory
    """
    tracking_address = output_dir+'/tb_logs' 
    with h5py.File(data_file,'r') as open_file :
        reco_data = np.array(open_file['smeared_data'])
        reco_met = np.array(open_file['smeared_met'])
        reco_ht = np.array(open_file['smeared_ht'])
        true_data = np.array(open_file['true_data'])
        true_met = np.array(open_file['true_met'])
        original_met = np.array(open_file['original_met'])
        true_ht = np.array(open_file['true_ht'])
        ids = np.array(open_file['ids'])
        ids_names = np.array(open_file['ids_names'])


    if variable=='original_met':
        graph_data = data_proc.METGraphCreator(reco_data,reco_met,reco_ht, original_met,true_ht,ids,log_features=log_features)
    elif variable=='true_met':
        graph_data = data_proc.METGraphCreator(reco_data,reco_met,reco_ht, true_met,true_ht,ids,log_features=log_features)
    elif variable=='true_ht':
        graph_data = data_proc.HTGraphCreator(reco_data,reco_ht,true_ht,ids,log_features=log_features)

    num_filters=1
    graph_conv_filters = graph_data.adjacency
    graph_conv_filters = K.constant(graph_conv_filters)
    
    hp = keras_tuner.HyperParameters()
    metrics = [nn_losses.MseThesholdMetric(threshold=t) for t in metric_thresholds]
    hypermodel = GraphAttentionHyperModel(features_input_shape=(graph_data.features.shape[1],graph_data.features.shape[2]), 
                                        adjancency_input_shape=(graph_data.adjacency.shape[1],graph_data.adjacency.shape[2]),
                                        filters_input_shape=(graph_conv_filters.shape[1],graph_conv_filters.shape[2]), 
                                        num_filters=num_filters, 
                                        loss_function=loss_function,
                                        metrics=metrics)


    tuner = keras_tuner.Hyperband(hypermodel = hypermodel,
                     objective = keras_tuner.Objective("val_loss", direction="min"),
                     max_epochs = max_epochs,
                     factor=hyperband_factor,
                     hyperband_iterations=1, #this should be as large as computationally possible. Default=1
                     seed=fixed_seed,
                     directory=output_dir,
                     project_name='hyperband_tuner')
    tuner.search_space_summary()
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-9),
        TensorBoard(log_dir=tracking_address, histogram_freq=1),
    ]
    tuner.search([graph_data.features, graph_data.adjacency,graph_conv_filters],graph_data.labels,
             validation_split=test_split,
             epochs=max_epochs,
             batch_size=batch_size,
             shuffle=True,
             callbacks=callbacks,
             use_multiprocessing=True,
             workers=3)

    tuner.results_summary()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Path to the input file ')
    parser.add_argument('--variable', type=str, help='Variable to train on : original_met, true_met, true_ht')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--log_features', type=str, help='Which features scale to be log')
    parser.add_argument('--metric_thresholds', type=str, help='List of metric thresholds to be monitored in MseThesholdMetric')
    parser.add_argument('--loss_function', type=str, help='Which loss function to use')
    parser.add_argument('--test_split', type=float, default = 0.2, help='test split')
    parser.add_argument('--batch_size', type=int, default = 1024, help='batch_size')
    parser.add_argument('--max_epochs', type=int, default = 50, help='Max epochs')
    parser.add_argument('--hyperband_factor', type=int, default = 3, help='Hyperband factor')

    args = parser.parse_args()
    args.loss_function = nn_losses.get_loss_func(args.loss_function)
    if args.log_features!='':
        args.log_features = [str(f) for f in args.log_features.replace(' ','').split(',')]
    else :
        args.log_features=[]    
    args.metric_thresholds = [float(f) for f in args.metric_thresholds.replace(' ','').split(',')]
    main_optimize_l1_teacher(**vars(args))




