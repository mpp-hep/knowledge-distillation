import math
import numpy as np
import random

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
    Concatenate,
    ZeroPadding2D,
    Reshape,
    Conv2D,
    AveragePooling2D,
    UpSampling2D,
    )
from qkeras import (
    QDense,
    QActivation,
    quantized_bits
    )

idx_met_0,idx_met_1=0,1
idx_eg_0,idx_eg_1=1,5
idx_mu_0,idx_mu_1=5,9
idx_jet_0,idx_jet_1=9,19

# number of integer bits for each bit width
QUANT_INT = {
    0: 0,
    2: 1,
    4: 2,
    6: 2,
    8: 3,
    10: 3,
    12: 4,
    14: 4,
    16: 6
    }


def kl_loss(y_true, y_pred):
    kl = y_true * (tf.math.log(y_true/y_pred))
    total_kl = tf.math.reduce_mean(kl)

    return total_kl


def smape(y_true, y_pred):

    return 100 * ( abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)) )


def mse_loss(inputs, outputs):

    return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)


def make_mse(inputs, outputs):

    # remove last dimension
    inputs = tf.squeeze(inputs, axis=-1)
    inputs = tf.cast(inputs, dtype=tf.float32)
    # trick with phi
    outputs_phi = math.pi*tf.math.tanh(outputs)
    # trick with phi
    outputs_eta_egamma = 3.0*tf.math.tanh(outputs)
    outputs_eta_muons = 2.1*tf.math.tanh(outputs)
    outputs_eta_jets = 4.0*tf.math.tanh(outputs)
    outputs_eta = tf.concat([outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    # use both tricks
    outputs = tf.concat([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # mask zero features
    mask = tf.math.not_equal(inputs,0)
    mask = tf.cast(mask, tf.float32)
    outputs = mask * outputs

    loss = mse_loss(tf.reshape(inputs, [-1, 57]), tf.reshape(outputs, [-1, 57]))
    loss = tf.math.reduce_mean(loss, axis=0) # average over batch

    return loss


class ModelWithShuffling(Model):
  def __init__(self, model, shuffle_strategy, shuffle_during):
    super().__init__()
    self.allowed_shuffle_strategies = 'shuffle_all,shuffle_within_pid,shuffle_within_between_pid'.split(',')
    self.shuffle_strategy = shuffle_strategy
    self.shuffle_during = shuffle_during
    self.main_model = model

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
            idx = tf.random.shuffle(tf.range(tot_particles))
            x_shuffled_block.append(tf.gather(x_block, tf.cast(idx, tf.int32), axis=1,))

        if len(x_shuffled_block)>1 :
            x_shuffled = Concatenate(axis=1)(x_shuffled_block)
        else :
            x_shuffled = x_shuffled_block
    return x_shuffled


  def call(self, inputs, training=False):
    x = inputs
    if (training and ('train' in self.shuffle_during)) or (not training and ('predict' in self.shuffle_during)):
        x = self.apply_shuffling(x)
    out = self.main_model(x)
    return out


def student_model(
    input_shape,
    node_size,
    quant_size,
    dropout,
    expose_latent:bool=False):

    int_size = QUANT_INT[quant_size]
    inp = Input(shape=(input_shape[1:]))
    if quant_size!=0:
        quantized_inputs = QActivation(f'quantized_bits(16,10,0,alpha=1)')(inp)
        x = Flatten()(quantized_inputs)
    else:
        quantized_inputs = None
        x = Flatten()(inp)
    x = BatchNormalization()(x)
    x = QDense(node_size*2,
        kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
        bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x) \
        if quant_size else \
        Dense(node_size*2)(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation(f'quantized_relu({quant_size},{int_size},0)')(x) \
        if quant_size else \
        Activation('relu')(x)
    x = QDense(node_size,
        kernel_quantizer = f'quantized_bits({quant_size},{int_size},0,alpha=1)',
        bias_quantizer = f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x) \
        if quant_size else \
        Dense(node_size)(x)
    latent = x
    if dropout:
        x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation(f'quantized_relu({quant_size},{int_size},0)')(x) \
        if quant_size else \
        Activation('relu')(x)
    x = QDense(node_size,
        kernel_quantizer = f'quantized_bits({quant_size},{int_size},0,alpha=1)',
        bias_quantizer = f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x) \
        if quant_size else \
        Dense(node_size)(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation(f'quantized_relu({quant_size},{int_size},0)')(x) \
        if quant_size else \
        Activation('relu')(x)
    x = QDense(1,
        kernel_quantizer = f'quantized_bits({quant_size},{int_size},0,alpha=1)',
        bias_quantizer = f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x) \
        if quant_size else \
        Dense(1)(x)
    # use higher precision for input and output
    out = QActivation(f'quantized_relu(16,10,0)')(x) \
        if quant_size else \
        Activation('relu')(x)
    if not expose_latent:
        main_model = Model(inputs=inp, outputs=out,name='main_model')
        main_model.summary()
        return main_model
    else:
        main_model = Model(inputs=inp, outputs=(latent, out),name='main_model')
        main_model.summary()
        return main_model


def teacher_model(
    image_shape,
    latent_dim,
    quant_size=0,
    pruning='not_pruned',
    expose_latent=False):

    int_size = QUANT_INT[quant_size]
    # encoder
    input_encoder = Input(shape=image_shape[1:], name='encoder_input')
    if quant_size!=0:
        quantized_inputs = QActivation(f'quantized_bits(16,10,0,alpha=1)')(input_encoder)
        x = ZeroPadding2D(((1,0),(0,0)))(quantized_inputs)
    else:
        quantized_inputs = None
        x = ZeroPadding2D(((1,0),(0,0)))(input_encoder)
    x = BatchNormalization()(x)
    #
    x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,3), use_bias=False, padding='valid',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = AveragePooling2D(pool_size=(3, 1))(x)
    #
    x = Flatten()(x)
    #
    enc = Dense(latent_dim)(x) if quant_size==0 \
        else QDense(latent_dim,
               kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
               bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)

    encoder = Model(inputs=input_encoder, outputs=enc)
    encoder.summary()
    # decoder
    input_decoder = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(64)(input_decoder) if quant_size==0 \
        else QDense(64,
               kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)',
               bias_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(input_decoder)
    #
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)
    #
    x = Reshape((2,1,32))(x)
    #
    x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(32, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((0,0),(1,1)))(x)

    x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(16, kernel_size=(3,1), use_bias=False, padding='same',
                         kernel_quantizer=f'quantized_bits({quant_size},{int_size},0,alpha=1)')(x)
    x = Activation('relu')(x) if quant_size==0 \
        else QActivation(f'quantized_relu({quant_size},{int_size},0)')(x)

    x = UpSampling2D((3,1))(x)
    x = ZeroPadding2D(((1,0),(0,0)))(x)

    dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x) if quant_size==0 \
        else QConv2D(1, kernel_size=(3,3), use_bias=False, padding='same',
                        kernel_quantizer='quantized_bits(16,10,0,alpha=1)')(x)
    #
    decoder = Model(inputs=input_decoder, outputs=dec)
    decoder.summary()

    if pruning=='pruned':
        start_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 5
        end_pruning = np.ceil(image_shape[0]*0.8/1024).astype(np.int32) * 15
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                initial_sparsity=0.0, final_sparsity=0.5,
                                begin_step=start_pruning, end_step=end_pruning)
        encoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(encoder, pruning_schedule=pruning_schedule)
        encoder = encoder_pruned
        decoder_pruned = tfmot.sparsity.keras.prune_low_magnitude(decoder, pruning_schedule=pruning_schedule)
        decoder = decoder_pruned

    # ae
    if expose_latent:
        latent = encoder(input_encoder)
        ae_outputs = decoder(latent)
        autoencoder = Model(inputs=input_encoder, outputs=(latent, ae_outputs))
    else:
        ae_outputs = decoder(encoder(input_encoder))
        autoencoder = Model(inputs=input_encoder, outputs=ae_outputs)
    autoencoder.summary()
    # load weights
    if pruning=='pruned':
        autoencoder = model_set_weights(autoencoder, f'output/model-conv_ae-8-b0-q0-not_pruned', quant_size)
    # compile AE
    autoencoder.compile(optimizer=Adam(lr=3E-3, amsgrad=True),
        loss=make_mse)

    return autoencoder


def student(
    image_shape,
    lr,
    dropout,
    node_size,
    distillation_loss,
    quant_size,
    particles_shuffle_strategy,
    particles_shuffle_during,
    expose_latent=False):

    model = ModelWithShuffling(model=student_model(image_shape, node_size, quant_size, dropout, expose_latent=expose_latent),
                               shuffle_strategy=particles_shuffle_strategy,
                               shuffle_during=particles_shuffle_during)
    # compile AE
    model.compile(optimizer=Adam(lr=lr, amsgrad=True),
        loss=distillation_loss)

    return model


def teacher(
    image_shape,
    lr,
    quant_size,
    particles_shuffle_strategy,
    particles_shuffle_during,
    expose_latent=False):

    model = ModelWithShuffling(model=teacher_model(image_shape, 8, quant_size=quant_size, expose_latent=expose_latent),
                               shuffle_strategy=particles_shuffle_strategy,
                               shuffle_during=particles_shuffle_during)
    # compile AE
    model.compile(optimizer=Adam(lr=lr, amsgrad=True),
        loss=make_mse)

    return model