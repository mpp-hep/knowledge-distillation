import os
import h5py
import pickle
import logging
import numpy as np
import keras_tuner
import tensorflow as tf
import argparse
import setGPU

from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperParameters
from tensorboard import program
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
    )

from models import make_mse
tracking_address = 'output/tb_logs' # the path of your log file for TensorBoard


class HyperTeacher(keras_tuner.HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self, hp):

        latent_dim = 8
        num_layers = hp.Choice('num_layers', values=[2, 3])

        if num_layers==3:
            first_conv2d = hp.Choice('conv2d', values=[256, 128])
            second_conv2d = int(first_conv2d/2)
            third_conv2d = int(second_conv2d/2)
        else:
            first_conv2d = hp.Choice('conv2d_1', values=[2*256, 2*128])
            second_conv2d = int(first_conv2d/2)

        # encoder
        input_encoder = keras.Input(shape=self.input_shape[1:], name='encoder_input')
        x = layers.ZeroPadding2D(((1,0),(0,0)))(input_encoder)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(first_conv2d, kernel_size=(3,3), use_bias=False, padding='valid')(x)
        x = layers.Activation('relu')(x)
        x = layers.AveragePooling2D(pool_size=(3, 1))(x)
        x = layers.Conv2D(second_conv2d, kernel_size=(3,1), use_bias=False, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.AveragePooling2D(pool_size=(3, 1))(x)
        if num_layers==3:
            x = layers.Conv2D(third_conv2d, kernel_size=(1,1), use_bias=False, padding='same')(x)
            x = layers.Activation('relu')(x)
            x = layers.AveragePooling2D(pool_size=(2, 1))(x)
        x = layers.Flatten()(x)
        enc = layers.Dense(latent_dim, name='latent_dense')(x)
        # decoder
        x = layers.Dense(third_conv2d if num_layers==3 else second_conv2d)(enc)
        x = layers.Activation('relu')(x)
        if num_layers==3:
            x = layers.Reshape((2,1,int(third_conv2d/2)))(x)
            x = layers.Conv2D(third_conv2d, kernel_size=(3,1), use_bias=False, padding='same')(x)
            x = layers.Activation('relu')(x)

        x = layers.Reshape((2,1,int(second_conv2d/2)))(x)
        x = layers.Conv2D(second_conv2d, kernel_size=(3,3), use_bias=False, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((3,1))(x)
        x = layers.ZeroPadding2D(((0,0),(1,1)))(x)
        x = layers.Conv2D(first_conv2d, kernel_size=(3,1), use_bias=False, padding='same')(x)
        x = layers.Activation('relu')(x)
        x = layers.UpSampling2D((3,1))(x)
        x = layers.ZeroPadding2D(((1,0),(0,0)))(x)
        dec = layers.Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x)
        # ae
        ae = keras.Model(input_encoder, dec, name='ae')
        ae.summary()
        # compile ae
        ae.compile(optimizer=Adam(lr=3E-3, amsgrad=True), loss=make_mse)

        return ae


def optimisation(args):

    # load data
    with open(args.input_file, 'rb') as f:
        _, _, x_test, y_test, _, _ = pickle.load(f)

    hypermodel = HyperTeacher(x_test.shape)
    tuner = keras_tuner.RandomSearch(
          hypermodel,
          objective='val_loss',
          max_model_size=500000,
          max_trials=20,
          overwrite=True,
          directory='output/hyper_tuning',
          )
    tuner.search_space_summary()
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[
        TensorBoard(log_dir=tracking_address, histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]
    tuner.search(
        x=x_test,
        y=y_test,
        epochs=50,
        batch_size=1024,
        validation_split=0.2,
        callbacks=callbacks
        )

    tuner.results_summary()
    logging.info('Get the optimal hyperparameters')

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    logging.info('Getting and printing best hyperparameters!')
    print(best_hps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('input_file', help='input file',
        type=str)

    args = parser.parse_args()
    optimisation(args)
