import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )
import pickle
import setGPU
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from models import student, teacher
from plot_results import BSM_SAMPLES

## just for testing things
config = {
    'learning_rate': 3e-3,
    'dropout': None,
    'node_size': 32,
    'distillation_loss': 'mae',
    'particles_shuffle_strategy': 'none',
    'particles_shuffle_during': 'never',
    'batch_size': 1024
}

idx_met_0,idx_met_1=0,1
idx_eg_0,idx_eg_1=1,5
idx_mu_0,idx_mu_1=5,9
idx_jet_0,idx_jet_1=9,19
log_loss=False

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

    lossm = mse_loss(tf.reshape(inputs, [-1, 57]), tf.reshape(outputs, [-1, 57]))
    loss = tf.math.reduce_mean(lossm, axis=0) # average over batch
    return loss, lossm

def reco_loss(inputs, outputs, dense=False):

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


mae = tf.keras.losses.MeanAbsoluteError()
mse = tf.keras.losses.MeanSquaredError()
cos = tf.keras.losses.CosineSimilarity()

@tf.function
def student_loss(true_loss, pred_loss):
    return mae(true_loss, pred_loss)

@tf.function
def teacher_loss(inputs, outputs):
    return make_mse(inputs, outputs)

def teacher_student_cotrain(data_file, 
                            student_output_result, 
                            teacher_output_train_loss, 
                            teacher_output_test_loss, 
                            teacher_output_signal_loss,
                            output_dir):
    
    # load the data
    with open(data_file, 'rb') as f:
        x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(f)
    
    # student model
    student_model = student(
        x_train.shape,
        config['learning_rate'],
        config['dropout'],
        config['node_size'],
        config['distillation_loss'],
        config['particles_shuffle_strategy'],
        config['particles_shuffle_during'],
        expose_latent=True
        )
    
    # teacher model 
    teacher_model = teacher(
        x_train.shape,
        3e-3,
        config['particles_shuffle_strategy'],
        config['particles_shuffle_during'],
        expose_latent=True
        )
    
    teacher_model.load_weights(f'{output_dir}teacher_ckpt/teacher_ckpt')
    student_model.load_weights(f'{output_dir}student_ckpt/student_ckpt')
    
    # prediction on training data
    _, predicted_loss = student_model.predict(x_train,batch_size=config['batch_size'])
    _, predicted_reco = teacher_model.predict(x_train,batch_size=config['batch_size'])
    _, actual_loss = teacher_loss(y_train, predicted_reco)
    print(actual_loss.shape)
    print(actual_loss)
    print(predicted_loss.shape)
    fig = plt.figure()
    plt.hist(np.log(1+predicted_loss),
            100,
            label='Student on training sample',
            linewidth=3,
            color='#016c59',
            histtype='step',
            density=True)
    plt.hist(np.log(1+actual_loss.numpy()),
            100,
            label='Teacher on training sample',
            linewidth=3,
            color='#7a5195',
            histtype='step',
            density=True)

    plt.semilogy()
    plt.ylabel('A.U.', )
    plt.xlabel('Loss on training sample', )
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, f'loss_on_training.png'))
    
    # prediction on test data
    _, predicted_loss = student_model.predict(x_test,batch_size=config['batch_size'])
    result_bsm = []
    for i, (bsm_data_name, bsm_id) in enumerate(zip(BSM_SAMPLES, [33,30,31,32])):
        bsm_data = all_bsm_data[i]
        _, predicted_bsm_data = student_model.predict(bsm_data,batch_size=config['batch_size'])
        result_bsm.append([bsm_data_name, np.log(1+predicted_bsm_data)])
    history = {}
    history['loss'] = [0]
    history['val_loss'] = [0]
    # save results
    with h5py.File(f'{student_output_result}', 'w') as h5f:
        if history: h5f.create_dataset('loss', data=history['loss'])
        if history: h5f.create_dataset('val_loss', data=history['val_loss'])
        h5f.create_dataset('predicted_loss', data=np.log(1+predicted_loss))
        for bsm in result_bsm:
            h5f.create_dataset(f'predicted_loss_{bsm[0]}', data=bsm[1])
            
    _, predicted_reco = teacher_model.predict(x_test,batch_size=config['batch_size'])
    _, actual_loss = teacher_loss(y_test, predicted_reco)
    
    with h5py.File(f'{teacher_output_test_loss}', 'w') as h5f:
        h5f.create_dataset('teacher_loss', data=np.log(1+actual_loss.numpy()))
        
    # test model on BSM data
    result_bsm = []
    for i, bsm_data_name in enumerate(BSM_SAMPLES):
        bsm_data = all_bsm_data[i]
        _, predicted_bsm_data = teacher_model.predict(bsm_data,batch_size=config['batch_size'])
        bsm_data = np.squeeze(bsm_data, axis=-1)
        bsm_data_target = np.copy(bsm_data)
        bsm_data_target[:,:,0] = pt_scaler.transform(bsm_data_target[:,:,0])
        bsm_data_target[:,:,0] = np.multiply(bsm_data_target[:,:,0], np.not_equal(bsm_data[:,:,0],0))
        bsm_data_target = bsm_data_target.reshape(bsm_data_target.shape[0],bsm_data_target.shape[1],bsm_data_target.shape[2],1)
        _, bsm_loss = teacher_loss(bsm_data_target, predicted_bsm_data)
        result_bsm.append([bsm_data_name, np.log(1+bsm_loss.numpy()), bsm_data_target])
     
    with h5py.File(f'{teacher_output_signal_loss}', 'w') as h5f:
        for i, bsm in enumerate(result_bsm):
            h5f.create_dataset(f'bsm_data_{bsm[0]}', data=all_bsm_data[i])
            h5f.create_dataset(f'teacher_loss_{bsm[0]}', data=bsm[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', type=str, help='Where is the data')
    parser.add_argument('--student-output-result', type=str, help='Student output file with results', required=True)
    parser.add_argument('--teacher-output-train-loss', type=str, help='Where is the data')
    parser.add_argument('--teacher-output-test-loss', type=str, help='Where is the data')
    parser.add_argument('--teacher-output-signal-loss', type=str, help='Where is the data')
    parser.add_argument('--output-dir', type=str, help='Where to put the other generated data')
    
    args = parser.parse_args()
    
    teacher_student_cotrain(**vars(args))