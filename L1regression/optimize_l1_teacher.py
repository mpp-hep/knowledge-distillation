import setGPU
import os
import argparse
import h5py
import numpy as np
import logging
import random
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
                            output_dir='',use_generator='',max_events=''):
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
        use_generator: bool, whether to use data generator or not
        max_events : int, max events to read (-1 : all)
    """
    tracking_address = output_dir+'/tb_logs' 
    with h5py.File(data_file,'r') as open_file :
        ids = np.array(open_file['ids'])
        ids_names = np.array(open_file['ids_names'])
        mask = ids>=0
        reco_data = np.array(open_file['smeared_data'])[mask]
        reco_met = np.array(open_file['smeared_met'])[mask]
        reco_ht = np.array(open_file['smeared_ht'])[mask]
        true_data = np.array(open_file['true_data'])[mask]
        true_met = np.array(open_file['true_met'])[mask]
        original_met = np.array(open_file['original_met'])[mask]
        true_ht = np.array(open_file['true_ht'])[mask]
        ids = np.array(open_file['ids'])[mask]
    
    emb_input_size = len(np.unique(reco_data[data_proc.idx_feature_for_met['pid']]))
    embedding_idx = data_proc.idx_feature_for_met['pid']
    if variable=='original_met':
        graph_data = data_proc.METGraphCreator(reco_data,reco_met,reco_ht, original_met,true_ht,ids,log_features=log_features)
    elif variable=='true_met':
        graph_data = data_proc.METGraphCreator(reco_data,reco_met,reco_ht, true_met,true_ht,ids,log_features=log_features)
    elif variable=='true_ht':
        emb_input_size = 0
        embedding_idx = -1
        graph_data = data_proc.HTGraphCreator(reco_data,reco_ht,true_ht,ids,log_features=log_features)


    graph_data.features, graph_data.adjacency,  graph_data.labels, graph_data.process_ids = graph_data.apply_mask_on_graph(graph_data.process_ids<2)
    
    num_filters=1
    graph_conv_filters = graph_data.adjacency

    #split the data in train and test:
    n_events = min(max_events, len(graph_data.features)) if max_events>0 else len(graph_data.features)
    indices = list(range(len(graph_data.features)))
    random.shuffle(indices)
    train_test_idx_split = int(len(graph_data.features)*test_split)
    val_indices = np.ones(len(indices), dtype=bool)
    val_indices[train_test_idx_split:] = False
    train_indices = np.ones(len(indices), dtype=bool)
    train_indices[:train_test_idx_split] = False
    train_indices[train_test_idx_split+int(n_events*(1-test_split)):] = False
    if use_generator:
        train_dataset = data_proc.DataGenerator(graph_data.features[train_indices], 
                                                graph_data.adjacency[train_indices],
                                                graph_conv_filters[train_indices],
                                                graph_data.labels[train_indices], 
                                                batch_size=batch_size,
                                                shuffle=True)
       #train_dataset = tf.data.Dataset.from_generator(data_proc.make_gen_callable(data_proc.DataGenerator(graph_data.features[train_indices], 
       #                                                                                                     graph_data.adjacency[train_indices],
       #                                                                                                     graph_conv_filters[train_indices],
       #                                                                                                     graph_data.labels[train_indices], 
       #                                                                                                     batch_size=batch_size)),
       #                                                 output_signature=((tf.TensorSpec(shape=( None,graph_data.features.shape[1],graph_data.features.shape[2]), dtype=tf.float32),
       #                                                 tf.TensorSpec(shape=( None,graph_data.adjacency.shape[1],graph_data.adjacency.shape[2]), dtype=tf.float32),
       #                                                 tf.TensorSpec(shape=( None,graph_conv_filters.shape[1],graph_conv_filters.shape[2]), dtype=tf.float32)),
       #                                                 tf.TensorSpec(shape=(None,2), dtype=tf.float32))
       #                                                 ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    else :
        train_dataset = ([graph_data.features[train_indices], graph_data.adjacency[train_indices],graph_conv_filters[train_indices]],graph_data.labels[train_indices])
    #Validation data always fits in the memory. Otherwise, change to generator as well
    val_dataset = ([graph_data.features[val_indices], graph_data.adjacency[val_indices],graph_conv_filters[val_indices]],graph_data.labels[val_indices])


    hp = keras_tuner.HyperParameters()
    metrics = [nn_losses.MseThesholdMetric(name=f'MseThesholdMetric_{t}',threshold=t) for t in metric_thresholds]
    hypermodel = GraphAttentionHyperModel(features_input_shape=(graph_data.features.shape[1],graph_data.features.shape[2]), 
                                        adjancency_input_shape=(graph_data.adjacency.shape[1],graph_data.adjacency.shape[2]),
                                        filters_input_shape=(graph_conv_filters.shape[1],graph_conv_filters.shape[2]), 
                                        emb_input_size=emb_input_size,
                                        embedding_idx=embedding_idx,
                                        num_filters=num_filters, 
                                        loss_function=loss_function,
                                        metrics=metrics)


    tuner = keras_tuner.Hyperband(hypermodel = hypermodel,
                     objective = keras_tuner.Objective("val_loss", direction="min"),
                     max_epochs = max_epochs,
                     factor=hyperband_factor,
                     hyperband_iterations=1, #this should be as large as computationally possible. Default=1
                   ### overwrite=False,
                     seed=fixed_seed,
                     directory=output_dir,
                     project_name='hyperband_tuner')
    ### RandomSearch is only used for testing
    #tuner = keras_tuner.RandomSearch(
    #      hypermodel = hypermodel ,
    #      objective=keras_tuner.Objective("val_loss", direction="min"),
    #      max_trials=1,
    #      directory=output_dir,
    #      project_name='hyperband_tuner',
    #      )

    tuner.search_space_summary()
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-9),
        TensorBoard(log_dir=tracking_address, histogram_freq=1),
    ]

    fit_dict = {'validation_data' : val_dataset,
             'batch_size' :batch_size,
             'shuffle' :False,
             'epochs':max_epochs,
             'callbacks' :callbacks,
             'use_multiprocessing' :False,
            # 'workers' :3
             }

    if use_generator : 
        tuner.search(train_dataset,**fit_dict) 
    else:
        tuner.search(train_dataset[0],train_dataset[1],**fit_dict) 

    tuner.results_summary()

    best_model = tuner.get_best_models()[0]
    best_model.build(graph_data.features.shape[1:])
    best_model.summary()
    
    if use_generator : 
        best_model.fit(train_dataset,**fit_dict) 
    else:
        best_model.fit(train_dataset[0],train_dataset[1],**fit_dict) 

    best_model.save(output_dir+'/best_model')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Path to the input file ')
    parser.add_argument('--variable', type=str, help='Variable to train on : original_met, true_met, true_ht')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--log_features', type=str,default='', help='Which features scale to be log')
    parser.add_argument('--metric_thresholds', type=str, help='List of metric thresholds to be monitored in MseThesholdMetric')
    parser.add_argument('--loss_function', type=str, help='Which loss function to use')
    parser.add_argument('--test_split', type=float, default = 0.2, help='test split')
    parser.add_argument('--batch_size', type=int, default = 1024, help='batch_size')
    parser.add_argument('--max_epochs', type=int, default = 50, help='Max epochs')
    parser.add_argument('--hyperband_factor', type=int, default = 3, help='Hyperband factor')
    parser.add_argument('--use_generator', type=int, default=0, help='True/False to use generator')
    parser.add_argument('--max_events', type=int, default = -1, help='Max events to train on')

    args = parser.parse_args()
    args.loss_function = nn_losses.get_loss_func(args.loss_function)
    if args.log_features!='':
        args.log_features = [str(f) for f in args.log_features.replace(' ','').split(',')]
    else :
        args.log_features=[]    
    args.use_generator = bool(args.use_generator)
    args.metric_thresholds = [float(f) for f in args.metric_thresholds.replace(' ','').split(',')]
    main_optimize_l1_teacher(**vars(args))




