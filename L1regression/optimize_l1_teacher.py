import os
import argparse
import numpy as np
import logging
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import keras_tuner
from keras_tuner import HyperParameters
from tensorboard import program
from tensorflow.keras.callbacks import (EarlyStopping,ReduceLROnPlateau,TensorBoard)
tracking_address = 'output/tb_logs' 
import utils.data_processing as data_proc
from nn.models import GraphAttentionHyperModel


def main_optimize_l1_teacher(data_file=''):
    """
    Performs optimization of the teacher model
    Arguments:
        data_file: str, path to the input file 
    """

    with h5py.File(data_file,'r') as open_file :
        reco_data = np.array(open_file['smeared_data'])
        reco_met = np.array(open_file['smeared_met'])
        reco_ht = np.array(open_file['smeared_ht'])
        true_data = np.array(open_file['true_data'])
        true_met = np.array(open_file['true_met'])
        true_ht = np.array(open_file['true_ht'])
        ids = np.array(open_file['ids'])
        ids_names = np.array(open_file['ids_names'])

    graph_data = data_proc.GraphCreator(reco_data,reco_met,reco_ht, true_met,true_ht,ids,log_features=['pt'])

    loss_function = 'mse' 
    num_filters=1
    graph_conv_filters = graph_data.adjacency
    graph_conv_filters = K.constant(graph_conv_filters)
    hp = keras_tuner.HyperParameters()
    hypermodel = GraphAttentionHyperModel(features_input_shape=(graph_data.features.shape[1],graph_data.features.shape[2]), 
                                        adjancency_input_shape=(graph_data.adjacency.shape[1],graph_data.adjacency.shape[2]),
                                        filters_input_shape=(graph_conv_filters.shape[1],graph_conv_filters.shape[2]), 
                                        num_filters=num_filters, 
                                        loss_function=loss_function)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Where is the data')
    args = parser.parse_args()
    main_optimize_l1_teacher(**vars(args))




