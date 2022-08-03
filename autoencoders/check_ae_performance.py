import math
import h5py
import os
import argparse
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

import matplotlib.pyplot as plt

from models import make_mse


def mse_loss(inputs, outputs):
    return np.mean(np.square(inputs-outputs), axis=-1)


def reco_loss(inputs, outputs):

    # trick on phi
    outputs_phi = math.pi*np.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0*np.tanh(outputs)
    outputs_eta_muons = 2.1*np.tanh(outputs)
    outputs_eta_jets = 4.0*np.tanh(outputs)
    outputs_eta = np.concatenate([outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    outputs = np.concatenate([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # calculate and apply mask
    mask = np.not_equal(inputs, 0)
    outputs = np.multiply(outputs, mask)

    reco_loss = mse_loss(inputs.reshape(-1,19*3), outputs.reshape(-1,19*3))
    return reco_loss


def evaluate(model, data_file, pt_scaler=None):

    if type(data_file)==str:
        with h5py.File(data_file,'r') as h5f:
            data = np.array(h5f['Particles'][:,:,:3])
            if len(data.shape)!=4:
                data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    else:
        data = data_file

    # get prediction
    predicted = model.predict(data)

    if pt_scaler:
        # test model on BSM data
        data = np.squeeze(data, axis=-1)
        data_target = np.copy(data)
        data_target[:,:,0] = pt_scaler.transform(data_target[:,:,0])
        data_target[:,:,0] = np.multiply(data_target[:,:,0], np.not_equal(data[:,:,0],0))
    else:
        data_target = data

    loss = reco_loss(data_target, predicted.astype(np.float32))

    return loss, data_target, predicted


def get_metric(background_loss, bsm_loss):

    target_val = np.concatenate((np.ones(bsm_loss.shape[0]), np.zeros(background_loss.shape[0])))
    predicted_val = np.concatenate((bsm_loss, background_loss))

    fpr, tpr, threshold_loss = roc_curve(target_val, predicted_val)
    auc_val = auc(fpr, tpr)

    return fpr, tpr, auc_val


def plot_rocs(background_loss, bsm_loss, bsm_name, color):

    fpr, tpr, auc = get_metric(background_loss, bsm_loss)
    plt.plot(
        fpr, tpr,
        '-',
        label=f'{bsm_name} (AUC = {auc*100:.0f}%)',
        linewidth=3,
        color=color)

    plt.xlim(10**(-6),1)
    plt.ylim(10**(-6),1.2)
    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3)
    plt.vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
    plt.legend(loc='lower right', frameon=False, title=f'ROC {bsm_name}')
    plt.tight_layout()


def plot_reconstruction(target, prediction, sample_name='bkg'):
    input_featurenames = ['pT', 'eta', 'phi']

    for i, feat in enumerate(input_featurenames):
        fig, axs = plt.subplots(1,3,figsize=(8,5))
        fig.suptitle('Kinematic distributions in data vs prediction')

        true0 = np.copy(target[:,1,i])
        predicted0 = np.copy(prediction[:,1,i])
        zeroes = [i for i,v in enumerate(true0) if v==0]
        true0 = np.delete(true0, zeroes)
        predicted0 = np.delete(predicted0, zeroes)
        if i==1:
            predicted0 = 3.0*np.tanh(predicted0)
        if i==2:
            predicted0 = math.pi*np.tanh(predicted0)
        axs[0].hist(true0,bins=100,label=r'Data',histtype='step', linewidth=2, facecolor='none', edgecolor='green',fill=True,density=True)
        axs[0].hist(predicted0,bins=100,label=r'Prediction',histtype='step', linewidth=2, facecolor='none', edgecolor='orchid',fill=True,density=True)
        axs[0].semilogy()
        axs[0].set(xlabel=f'Leading e/$\gamma$ {feat} ( Norm. GeV)', ylabel='A.U')
        axs[0].legend(loc='best',frameon=False, ncol=1, fontsize='large')

        true0 = np.copy(target[:,5,i])
        predicted0 = np.copy(prediction[:,5,i])
        zeroes = [i for i,v in enumerate(true0) if v==0]
        true0 = np.delete(true0, zeroes)
        predicted0 = np.delete(predicted0, zeroes)
        if i==1:
            predicted0 = 2.1*np.tanh(predicted0)
        if i==2:
            predicted0 = math.pi*np.tanh(predicted0)
        axs[1].hist(true0,bins=100,label=r'Data',histtype='step', linewidth=2, facecolor='none', edgecolor='green',fill=True,density=True)
        axs[1].hist(predicted0,bins=100,label=r'Prediction',histtype='step', linewidth=2, facecolor='none', edgecolor='orchid',fill=True,density=True)
        axs[1].set(xlabel=f'Leading muon {feat} (Norm. GeV)', ylabel='A.U')
        axs[1].semilogy()
        axs[1].legend(loc='best',frameon=False, ncol=1, fontsize='large')

        true0 = np.copy(target[:,9,i])
        predicted0 = np.copy(prediction[:,9,i])
        zeroes = [i for i,v in enumerate(true0) if v==0]
        true0 = np.delete(true0, zeroes)
        predicted0 = np.delete(predicted0, zeroes)
        if i==1:
            predicted0 = 4.0*np.tanh(predicted0)
        if i==2:
            predicted0 = math.pi*np.tanh(predicted0)
        axs[2].hist(true0,bins=100,label=r'Data',histtype='step', linewidth=2, facecolor='none', edgecolor='green',fill=True,density=True)
        axs[2].hist(predicted0,bins=100,label=r'Prediction',histtype='step', linewidth=2, facecolor='none', edgecolor='orchid',fill=True,density=True)
        axs[2].set(xlabel=f'Leading jet {feat} (Norm. GeV)', ylabel='A.U')
        axs[2].semilogy()
        axs[2].legend(loc='best',frameon=False, ncol=1, fontsize='large')
        plt.savefig(f'plots/reco_{sample_name}_{feat}.pdf')
        plt.clf()


def main(args):
    """
     if only the trained model is passed
     first evaluate the trained model on the signals
     else, go directly to plotting
     load teacher model
    """
    if args.pretrained_ae_json:
        with open(args.pretrained_ae_json, 'r') as jsonfile:
            config = jsonfile.read()
        model = tf.keras.models.model_from_json(
            config,
            custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude}
            )
        model.load_weights(args.pretrained_ae_h5)
        model.summary()
    else:
        model = tf.keras.models.load_model(
            args.pretrained_ae_h5,
            custom_objects={'make_mse': make_mse}
            )

    # load dataset and pt scaler
    with open(args.dataset, 'rb') as f:
        x_train, y_train, x_test, y_test, _, _, _, pt_scaler, _, _, _, _ = pickle.load(f)

    background_loss, x_target, x_predicted  = evaluate(model, x_test, pt_scaler)

    colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']

    for i, bsm in enumerate(args.signal):
        bsm_loss, _, _ = evaluate(model, bsm[1], pt_scaler)
        plot_rocs(background_loss, bsm_loss, bsm[0], colors[i])

    plt.savefig(f'plots/rocs.pdf')
    print('Saved your plots in plots/rocs.pdf')
    plt.clf()

    # plot reconstructions
    plot_reconstruction(x_target, x_predicted)
    for i, bsm in enumerate(args.signal):
        bsm_loss, bsm_target, bsm_predicted = evaluate(model, bsm[1], pt_scaler)
        plot_reconstruction(bsm_target, bsm_predicted, bsm[0])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', help='Path to the datast',
        type=str)

    parser.add_argument('--pretrained-ae-json', help='Path to the saved model json',
        type=str, default=None)
    parser.add_argument('--pretrained-ae-h5', help='Path to the saved model h5',
        type=str, default=None)
    parser.add_argument('--signal', help='Name of the BSM model and path to the dataset',
        action='append', nargs='+', default=None)
    parser.add_argument('--pt-scaler', help='Path to the saved pt scaler',
        type=str, default=None)

    args = parser.parse_args()
    main(args)
