import numpy as np
import random
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
    Activation,
    Lambda
    )
from qkeras import (
    QDense,
    QActivation,
    quantized_bits
    )
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from reformat_ae_l1_data_shuffling import idx_met_0,idx_met_1,idx_eg_0,idx_eg_1,idx_mu_0,idx_mu_1,idx_jet_0,idx_jet_1


def kl_loss(y_true, y_pred):
    kl = y_true * (tf.math.log(y_true/y_pred))
    total_kl = tf.math.reduce_mean(kl)
    return total_kl

def smape(y_true, y_pred):
    return 100 * ( abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)) )




class StudentModel(tf.keras.Model):
  def __init__(self,image_shape, dropout, node_size, shuffle_strategy, shuffle_during, quantize):
    super().__init__()
    self.allowed_shuffle_strategies = 'shuffle_all,shuffle_within_pid,shuffle_within_between_pid'.split(',')
    self.shuffle_strategy = shuffle_strategy
    self.shuffle_during = shuffle_during
    self.image_shape = image_shape
    self.dropout = dropout
    self.node_size = node_size 
    self.quantize = quantize
    self.main_model = self.build_main_model()


  def apply_shuffling(self,x):
    particles_blocks = []
    x_shuffled_block = []
    if self.shuffle_strategy not in self.allowed_shuffle_strategies:
        x_shuffled = x
    else:
        if self.shuffle_strategy=='shuffle_all':
            particles_blocks.append([0,tf.shape(x)[1]])
        if self.shuffle_strategy=='shuffle_within_pid' or self.shuffle_strategy=='shuffle_within_between_pid':
            for start,end in zip([idx_met_0,idx_eg_0,idx_mu_0,idx_jet_0],[idx_met_1,idx_eg_1,idx_mu_1,idx_jet_1]):
                particles_blocks.append([start,end])
        if self.shuffle_strategy=='shuffle_within_between_pid':
            random.shuffle(particles_blocks) #shuffle the order of PID blocks                

        for block in particles_blocks:
            tot_particles = block[1]-block[0]
            x_block = x[:,block[0]:block[1],:]
            idx = Lambda(lambda y: tf.random.shuffle(tf.range(y)))(tot_particles)
            x_shuffled_block.append(Lambda(lambda y: tf.gather(y[0], tf.cast(y[1], tf.int32), axis=1,))([x_block, idx]))
            
        if len(x_shuffled_block)>1 :
            x_shuffled = tf.keras.layers.Concatenate(axis=1)(x_shuffled_block)
        else :
            x_shuffled = x_shuffled_block
    return x_shuffled

  def build_main_model(self):
    inp = Input(shape=(self.image_shape[1:]))
    x = Flatten()(inp)
    x = BatchNormalization()(x)
    x = QDense(self.node_size*2,
        kernel_quantizer="quantized_bits(6,0,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,0,alpha=1)")(x) \
        if self.quantize else \
        Dense(self.node_size*2)(x)
    if self.dropout:
        x = Dropout(self.dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if self.quantize else \
        Activation('relu')(x)
    x = QDense(self.node_size,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if self.quantize else \
        Dense(self.node_size)(x)
    if self.dropout:
        x = Dropout(self.dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if self.quantize else \
        Activation('relu')(x)
    x = QDense(self.node_size,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if self.quantize else \
        Dense(self.node_size)(x)
    if self.dropout:
        x = Dropout(self.dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if self.quantize else \
        Activation('relu')(x)
    x = QDense(1,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if self.quantize else \
        Dense(1)(x)
    out = QActivation("quantized_relu(6,0)")(x) \
        if self.quantize else \
        Activation('relu')(x)
    main_model = Model(inputs=inp, outputs=out,name='main_model')
    main_model.summary()
    return main_model

        
  def call(self, inputs, training=False):
    x = inputs
    if (training and ('train' in self.shuffle_during)) or (not training and ('predict' in self.shuffle_during)):
        #tf.print("before:", x, summarize=-1)
        x = self.apply_shuffling(x)
        #tf.print("after:", x, summarize=-1)
    out = self.main_model(x)
    return out



def student(image_shape, lr, dropout, node_size, distillation_loss,particles_shuffle_strategy,particles_shuffle_during):
    quantize = False
    model = StudentModel(image_shape=image_shape,
                         dropout=dropout,
                         node_size=node_size, 
                         shuffle_strategy=particles_shuffle_strategy,
                         shuffle_during=particles_shuffle_during,
                         quantize=quantize)
    # compile AE
    model.compile(optimizer=Adam(lr=lr, amsgrad=True), 
        loss=distillation_loss)
    return model
