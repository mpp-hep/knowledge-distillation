import os
import h5py
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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


class HyperTeacher(keras_tuner.HyperModel):

    def __init__(self, input_shape):
        self.input_shape = input_shape


    def build(self, hp):

        latent_dim = 8

        # encoder
        input_encoder = Input(shape=self.input_shape[1:], name='encoder_input')
        x = ZeroPadding2D(((1,0),(0,0)))(input_encoder)
        x = BatchNormalization()(x)
        x = Conv2D(16, kernel_size=(3,3), use_bias=False, padding='valid')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3, 1))(x)
        x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(3, 1))(x)
        x = Flatten()(x)
        enc = Dense(latent_dim, name='latent_dense')(x)

        encoder = Model(inputs=input_encoder, outputs=enc, name='encoder_CNN')
        encoder.summary()

        # decoder
        input_decoder = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(64)(input_decoder)
        x = Activation('relu')(x)
        x = Reshape((2,1,32))(x)
        x = Conv2D(32, kernel_size=(3,1), use_bias=False, padding='same')(x)
        x = Activation('relu')(x)
        x = UpSampling2D((3,1))(x)
        x = ZeroPadding2D(((0,0),(1,1)))(x)
        x = Conv2D(16, kernel_size=(3,1), use_bias=False, padding='same')(x)
        x = Activation('relu')(x)
        x = UpSampling2D((3,1))(x)
        x = ZeroPadding2D(((1,0),(0,0)))(x)
        dec = Conv2D(1, kernel_size=(3,3), use_bias=False, padding='same')(x)
        decoder = Model(inputs=input_decoder, outputs=dec)
        decoder.summary()
        # vae
        vae_outputs = decoder(encoder(input_encoder)[2])
        vae = Model(input_encoder, vae_outputs, name='vae')
        vae.summary()
        # compile VAE
        vae.compile(
            optimizer=Adam(
                lr=3E-3,
                amsgrad=True),
            loss=make_mse
            )

        return vae

        # inputs = keras.Input(shape=self.input_shape[1:])
        # x = layers.Flatten()(inputs)
        # # Number of hidden layers of the MLP is a hyperparameter.
        # for i in range(hp.Int('mlp_layers', 2, 4)):
        #     # Number of units of each layer are
        #     # different hyperparameters with different names.
        #     output_node = layers.Dense(
        #         units=hp.Choice(f'units_{i}', [4, 8, 16, 32], default=32),
        #         activation='relu',
        #     )(x)

        # # The last layer contains 1 unit, which
        # # represents the learned loss value
        # outputs = layers.Dense(units=1, activation='relu')(x)
        # hyper_student = keras.Model(inputs=inputs, outputs=outputs)

        # hyper_student.compile(
        #     optimizer=Adam(lr=3E-3, amsgrad=True),
        #     loss=self.distillation_loss
        #     )


def optimisation(args):

    # load teacher's loss for training
    with h5py.File(args.input_file, 'r') as f:
        x_train = np.array(f['data'][:,:,:3])

    hypermodel = HyperTeacher(x_train.shape)
    tuner = keras_tuner.RandomSearch(
          hypermodel,
          objective='val_loss',
          max_trials=len(hypermodel.model_configurations),
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
        x=x_train,
        y=y_train,
        epochs=1,
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
