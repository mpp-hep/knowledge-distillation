import os
import h5py
import logging
import numpy as np
import kerastuner
import tensorflow as tf
import argparse
import setGPU

from tensorflow.keras.optimizers import Adam
from kerastuner import HyperParameters

from models import student_model


class HyperStudent(kerastuner.HyperModel):

    def __init__(self, input_shape, distillation_loss):
        self.input_shape = input_shape
        self.distillation_loss = distillation_loss


    def build(self, hp):

        filters = hp.Choice('filters', [4, 8, 16, 32], default=4)

        hyper_student = student_model(
            self.input_shape,
            filters,
            quant_size=0,
            dropout=None,
            expose_latent=False
            )

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
    tuner = kerastuner.RandomSearch(
          hypermodel,
          objective='val_loss',
          max_trials=10,
          overwrite=True,
          directory='output/hyper_tuning',
          )
    tuner.search_space_summary()
    tuner.search(
        x=x_train,
        y=y_train,
        epochs=1,
        validation_split=0.2
        )

    tuner.results_summary()
    logging.info('Get the optimal hyperparameters')

    best_hps = tuner.get_best_hyperparameters(num_trials=5)[0]

    logging.info('Getting and printing best hyperparameters!')
    print(best_hps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--distillation-loss', type=str, default='mse', help='Loss to use for distillation')
    args = parser.parse_args()
    optimisation(**vars(args))