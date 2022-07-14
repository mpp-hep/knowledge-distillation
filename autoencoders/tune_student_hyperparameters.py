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

tracking_address = '../output/tb_logs' # the path of your log file for TensorBoard

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class HyperStudent(keras_tuner.HyperModel):

    def __init__(self, input_shape, distillation_loss, param_threshold=(4500,5000)):
        self.input_shape = input_shape
        self.distillation_loss = distillation_loss

        self.num_layers = [2,3,4]
        self.num_params = [4,8,16,32, 64]

        model_configurations = []
        self.model_configurations = []

        for nl in self.num_layers:
            grid_choices = np.tile(self.num_params, (nl,1))
            configs = np.array(np.meshgrid(*grid_choices)).T.reshape(-1, nl)

            model_configurations.append(configs.tolist())

        model_configurations = [num for sublist in model_configurations for num in sublist]
        for config in model_configurations:
            params = self.compute_model_params(config)
            if params <= param_threshold[1] and params >= param_threshold[0]:
                self.model_configurations.append(config)
        print('Total feasible configurations: ', len(self.model_configurations))

    def compute_model_params(self, config):
        total_params = 0
        total_params += np.prod(self.input_shape[1:])*config[0]
        total_params += config[-1]
        for i in range(len(config)-1):
            total_params += config[i]*config[i+1]
        return total_params 


    def build(self, hp):

        inputs = keras.Input(shape=self.input_shape[1:])
        x = layers.Flatten()(inputs)

        config_index = hp.Int("config_indx", min_value=0, max_value=len(self.model_configurations)-1, step=1)

        # Number of hidden layers of the MLP is a hyperparameter.
        for units in self.model_configurations[config_index]:
            # Number of units of each layer are
            # different hyperparameters with different names.
            x = layers.Dense(units=units,activation='relu')(x)

        # The last layer contains 1 unit, which
        # represents the learned loss value
        outputs = layers.Dense(units=1, activation='relu')(x)
        hyper_student = keras.Model(inputs=inputs, outputs=outputs)

        hyper_student.compile(
            optimizer=Adam(lr=3E-3, amsgrad=True),
            loss=self.distillation_loss
            )

        return hyper_student

def optimisation(input_file, distillation_loss):

    # load teacher's loss for training
    with h5py.File(input_file, 'r') as f:
        x_train = np.array(f['data'][:,:,:3])
        y_train = np.array(f['teacher_loss'])

    hypermodel = HyperStudent(x_train.shape, distillation_loss)
    tuner = keras_tuner.RandomSearch(
          hypermodel,
          objective='val_loss',
          max_trials=1,
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
    print('Optimal Configuration:', hypermodel.model_configurations[best_hps['config_indx']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--distillation-loss', type=str, default='mse', help='Loss to use for distillation')
    args = parser.parse_args()
    optimisation(**vars(args))