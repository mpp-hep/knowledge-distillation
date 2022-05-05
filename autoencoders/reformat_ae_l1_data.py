import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, QActivation
import pickle
import setGPU

from plotting import reco_loss, BSM_SAMPLES

def reformat_ae_l1_data(data_file, teacher_input_json, teacher_input_h5,
    output_train_loss, output_test_loss, output_signal_loss):

    # load data
    with open(data_file, 'rb') as f:
        x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(f)

    # load teacher model
    with open(teacher_input_json, 'r') as jsonfile:
        config = jsonfile.read()
    teacher_model = tf.keras.models.model_from_json(config,
        custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
            'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
    teacher_model.load_weights(teacher_input_h5)
    teacher_model.summary()

    y_teacher_train = reco_loss(y_train, teacher_model.predict(x_train))
    with h5py.File(output_train_loss, 'w') as h5f:
        h5f.create_dataset('teacher_loss', data=y_teacher_train)
        h5f.create_dataset('data', data=x_train)

    y_teacher_test = reco_loss(y_test, teacher_model.predict(x_test))
    with h5py.File(output_test_loss, 'w') as h5f:
        h5f.create_dataset('teacher_loss', data=y_teacher_test)
        h5f.create_dataset('data', data=x_test)

    # test model on BSM data
    result_bsm = []
    for i, bsm_data_name in enumerate(BSM_SAMPLES):
        bsm_data = all_bsm_data[i]
        predicted_bsm_data = teacher_model.predict(bsm_data)
        bsm_data = np.squeeze(bsm_data, axis=-1)
        bsm_data_target = np.copy(bsm_data)
        bsm_data_target[:,:,0] = pt_scaler.transform(bsm_data_target[:,:,0])
        bsm_data_target[:,:,0] = np.multiply(bsm_data_target[:,:,0], np.not_equal(bsm_data[:,:,0],0))
        bsm_data_target = bsm_data_target.reshape(bsm_data_target.shape[0],bsm_data_target.shape[1],bsm_data_target.shape[2],1)
        result_bsm.append([bsm_data_name, reco_loss(bsm_data_target, predicted_bsm_data)])

    with h5py.File(output_signal_loss, 'w') as h5f:
        for i, bsm in enumerate(result_bsm):
            h5f.create_dataset(f'bsm_data_{bsm[0]}', data=all_bsm_data[i])
            h5f.create_dataset(f'teacher_loss_{bsm[0]}', data=bsm[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, help='Where is the data')
    parser.add_argument('--teacher-input-json', type=str, help='Where is the data')
    parser.add_argument('--teacher-input-h5', type=str, help='Where is the data')
    parser.add_argument('--output-train-loss', type=str, help='Where is the data')
    parser.add_argument('--output-test-loss', type=str, help='Where is the data')
    parser.add_argument('--output-signal-loss', type=str, help='Where is the data')
    args = parser.parse_args()
    reformat_ae_l1_data(**vars(args))
