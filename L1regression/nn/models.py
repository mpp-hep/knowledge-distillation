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
    'Graph Attention Hyper Model '
    def __init__(self, features_input_shape, adjancency_input_shape, filters_input_shape,emb_input_size, embedding_idx, num_filters, loss_function,metrics=[]):
        self.features_input_shape = features_input_shape
        self.adjancency_input_shape = adjancency_input_shape
        self.filters_input_shape = filters_input_shape
        self.emb_input_size = emb_input_size
        self.embedding_idx = embedding_idx
        self.num_filters = num_filters
        self.loss_function = loss_function
        self.metrics = metrics

    def build(self, hp):
        features_input = keras.Input(shape=self.features_input_shape, name='features_input')
        adjancency_input = keras.Input(shape=self.adjancency_input_shape, name='adjancency_input')
        filters_input = keras.Input(shape=self.filters_input_shape, name='filters_input')
        activation=hp.Choice("activation", ["relu", "elu"])

        if self.embedding_idx>=0 :
            feat_idx = np.arange(self.features_input_shape[1],dtype=int)
            feat_idx = np.delete(feat_idx, [self.embedding_idx], axis=0)
            x_emb =  features_input[:,:,self.embedding_idx]
            x_feats = tf.gather(features_input,tf.constant(feat_idx),axis=-1)
            x_emb = keras.layers.Embedding(input_dim=self.emb_input_size,
                                output_dim = hp.Int("embedding" , min_value=2, max_value=4, step=1),
                                embeddings_regularizer=l2(5e-6)
                                )(x_emb)  
            x = keras.layers.Concatenate(axis=-1)([x_feats, x_emb])
        else :
            x=features_input

        x = keras.layers.BatchNormalization()(x)
        
        for i in range(1, hp.Int("num_layers", 3, 5)): # 4 7
            x = MultiGraphAttentionCNN(output_dim=hp.Int("units_" + str(i), min_value=64, max_value=128, step=32), #32 , 256, 32
                 num_filters=self.num_filters, 
                 num_attention_heads=hp.Int("heads_" + str(i), min_value=3, max_value=5, step=1), 
                 attention_combine='concat', 
                 attention_dropout=hp.Float("attention_dropout_" + str(i), 0, 0.05, step=0.02), 
                 activation=activation, 
                 kernel_regularizer=l2(5e-6))([x, adjancency_input, filters_input])

            x = keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.05, step=0.02))(x)

        x = keras.layers.Lambda(lambda x: K.sum(x, axis=1))(x)  # adding a node invariant layer (sum/mean/max/min) to make sure output does not depends upon the node order in a graph.
        output = keras.layers.Dense(1, activation=activation)(x)
        model = Model(inputs=[features_input, adjancency_input, filters_input], outputs=output, name='graph_att_model')
    
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=self.loss_function,
                  metrics=self.metrics)
    
        return model







