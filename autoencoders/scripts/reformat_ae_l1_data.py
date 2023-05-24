import math
import argparse
import h5py
import numpy as np
import pickle

import tensorflow as tf
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import (
    QConv2D,
    QDense,
    QActivation
    )

from plot_results import BSM_SAMPLES
from models import (
    make_mse,
    idx_met_0,
    idx_met_1,
    idx_eg_0,
    idx_eg_1,
    idx_mu_0,
    idx_mu_1,
    idx_jet_0,
    idx_jet_1
    )


def mse_loss(inputs, outputs):

    return np.mean(np.square(inputs-outputs), axis=-1)


def reco_loss(
        inputs,
        outputs,
        dense=False):

    if dense:
        outputs = outputs.reshape(outputs.shape[0],19,3,1)
        inputs = inputs.reshape(inputs.shape[0],19,3,1)

    # trick on phi
    outputs_phi = math.pi*np.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0*np.tanh(outputs)
    outputs_eta_muons = 2.1*np.tanh(outputs)
    outputs_eta_jets = 4.0*np.tanh(outputs)
    outputs_eta = np.concatenate([outputs[:,idx_met_0:idx_met_1,:,:], outputs_eta_egamma[:,idx_eg_0:idx_eg_1,:,:], outputs_eta_muons[:,idx_mu_0:idx_mu_1,:,:], outputs_eta_jets[:,idx_jet_0:idx_jet_1,:,:]], axis=1)
    outputs = np.concatenate([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # change input shape
    inputs = np.squeeze(inputs, -1)
    # # calculate and apply mask
    mask = np.not_equal(inputs, 0)
    outputs = np.multiply(outputs, mask)

    reco_loss = mse_loss(inputs.reshape(inputs.shape[0],57), outputs.reshape(outputs.shape[0],57))

    return reco_loss


def main(args):

    # load teacher model
    if args.teacher_input_json:
        with open(args.teacher_input_json, 'r') as jsonfile:
            config = jsonfile.read()
        teacher_model = tf.keras.models.model_from_json(config,
            custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
                'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
        teacher_model.load_weights(args.teacher_input_h5)
        teacher_model.summary()
    else:
        teacher_model = tf.keras.models.load_model(
            args.teacher_input_h5,
            custom_objects={'make_mse': make_mse}
            )
    # load datasets
    datasets = np.load(args.data_file)

    y_teacher_train = reco_loss(datasets['y_train'], teacher_model.predict(datasets['x_train']))
    y_teacher_test = reco_loss(datasets['y_test'], teacher_model.predict(datasets['x_test']))
    y_teacher_val = reco_loss(datasets['y_val'], teacher_model.predict(datasets['x_val']))
    np.savez(args.output_loss,
        teacher_train_loss=np.log(y_teacher_train+1) if args.log_loss else y_teacher_train,
        x_train=datasets['x_train'],
        teacher_test_loss=np.log(y_teacher_test+1) if args.log_loss else y_teacher_test,
        x_test=datasets['x_test'],
        teacher_val_loss=np.log(y_teacher_val+1) if args.log_loss else y_teacher_val,
        x_val=datasets['x_val'])

    # load data
    with open(args.pt_scaler_file, 'rb') as f:
        pt_scaler = pickle.load(f)
    bsm_datasets = np.load(args.bsm_file)

    # test model on BSM data
    signal_dict = dict()
    for bsm_data_name, bsm_data in enumerate(bsm_datasets):
        predicted_bsm_data = teacher_model.predict(bsm_data)
        bsm_data = np.squeeze(bsm_data, axis=-1)
        bsm_data_target = np.copy(bsm_data)
        bsm_data_target[:,:,0] = pt_scaler.transform(bsm_data_target[:,:,0])
        bsm_data_target[:,:,0] = np.multiply(bsm_data_target[:,:,0], np.not_equal(bsm_data[:,:,0], 0))
        bsm_data_target = bsm_data_target.reshape(bsm_data_target.shape[0], bsm_data_target.shape[1], bsm_data_target.shape[2], 1)
        bsm_loss = reco_loss(bsm_data_target, predicted_bsm_data)

        signal_dict[f'bsm_data_{bsm_data_name}'] = bsm_data
        signal_dict[f'teacher_loss_{bsm_data_name}'] = data=np.log(bsm_loss+1) if args.log_loss else bsm_loss

    np.savez(args.output_signal_loss, **signal_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('data_file', type=str,
        help='Path to the input datasets file')
    parser.add_argument('bsm_file', type=str,
        help='Path to the input BSM datasets file')
    parser.add_argument('teacher_input_h5', type=str,
        help='Path to pre-trained teacher model')

    parser.add_argument('--teacher-input-json', type=str,
        help='If teacher saved as both h5 and json', default=None)
    parser.add_argument('--output-loss', type=str,
        help='Where to save the loss')
    parser.add_argument('--output-signal-loss', type=str,
        help='Where to save the loss')
    parser.add_argument('--log-loss', type=bool, default=False,
        help='Apply log to the loss or not : True/False')
    parser.add_argument('--pt-scaler', help='Path to the saved pt scaler',
        type=str, default=None)

    args = parser.parse_args()
    main(args)
