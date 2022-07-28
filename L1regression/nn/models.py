import os
import argparse
import numpy as np
import keras_tuner
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import keras_dgl
from keras_dgl.layers import GraphAttentionCNN,MultiGraphCNN,MultiGraphAttentionCNN


class GraphAttentionHyperModel(keras_tuner.HyperModel):
    def __init__(self, features_input_shape, adjancency_input_shape, filters_input_shape, num_filters, loss_function):
        self.features_input_shape = features_input_shape
        self.adjancency_input_shape = adjancency_input_shape
        self.filters_input_shape = filters_input_shape
        self.num_filters = num_filters
        self.loss_function = loss_function

    def build(self, hp):
        # Initialize sequential API and start building model.
        features_input = keras.Input(shape=self.features_input_shape)
        adjancency_input = keras.Input(shape=self.adjancency_input_shape)
        filters_input = keras.Input(shape=self.filters_input_shape)
        activation=hp.Choice("activation", ["relu", "elu"])

        x=features_input

        # Tune the number of hidden layers and units in each.
        # Number of hidden layers: 3 - 6
        # Number of Units: 32 - 256 with stepsize of 32

        x = keras.layers.BatchNormalization()(x)

        for i in range(1, hp.Int("num_layers", 4, 7)):
            x = MultiGraphAttentionCNN(output_dim=hp.Int("units_" + str(i), min_value=32, max_value=256, step=32),
                 num_filters=self.num_filters, 
                 num_attention_heads=hp.Int("heads_" + str(i), min_value=3, max_value=5, step=1), 
                 attention_combine='concat', 
                 attention_dropout=hp.Float("attention_dropout_" + str(i), 0, 0.3, step=0.1), 
                 activation=activation, 
                 kernel_regularizer=l2(5e-4))([x, adjancency_input, filters_input])

            # Tune dropout layer with values from 0 - 0.3 with stepsize of 0.1.
            x = keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1))(x)

        x = keras.layers.Lambda(lambda x: K.mean(x, axis=1))(x)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
        output = keras.layers.Dense(1, activation=activation)(x)
        model = Model(inputs=[features_input, adjancency_input, filters_input], outputs=output)
    
        # Tune learning rate for Adam optimizer with values from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
        # Define optimizer, loss, and metrics
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=self.loss_function)
    
        return model

    #TODO : Check if fit function is needed
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            # Tune whether to shuffle the data in each epoch.
            shuffle=hp.Boolean("shuffle"), #TODO : check this
            **kwargs,
        )







