import os
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
import json

import matplotlib.pyplot as plt

from models import student
from plot_results import BSM_SAMPLES


def knowledge_distillation(input_train_file, input_test_file, input_signal_file,
    data_name, n_features, teacher_loss_name, output_model_h5, output_model_json,
    output_history, batch_size, n_epochs, distillation_loss, dropout,
    learning_rate, node_size, output_result, output_dir,
    particles_shuffle_strategy,particles_shuffle_during):

    # load teacher's loss for training
    with h5py.File(input_train_file, 'r') as f:
        x_train = np.array(f[data_name][:,:,:n_features])
        y_train = np.array(f[teacher_loss_name])


    # student model
    student_model = student(
        x_train.shape,
        learning_rate,
        dropout,
        node_size,
        distillation_loss,
        particles_shuffle_strategy,
        particles_shuffle_during
        )

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]

    # train student to reproduce teachers' loss
    print('Starting training')
    history = student_model.fit(x=x_train, y=y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks)

    plt.hist(student_model.predict(x_train,batch_size=batch_size),
            100,
            label='Student on training sample',
            linewidth=3,
            color='#016c59',
            histtype='step',
            density=True)
    plt.hist(y_train,
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
    nametag = output_model_h5[output_model_h5.find('model_')+len('model_'):output_model_h5.find('.h5')]
    plt.savefig(os.path.join(output_dir, f'loss_on_training_'+nametag+'.pdf'))

    # save student model
    #TO DO: With custom model it is not trivial to save model as a json. Either fix or remove save-json option 
    #student_model_json = student_model.to_json()
    with open(output_model_json, 'w') as json_file:
        #json_file.write(student_model_json)
        json_file.write(json.dumps({}))
    student_model.save_weights(output_model_h5)

    # save training history
    if history:
        with open(output_history, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load testing set
    with h5py.File(input_test_file, 'r') as f:
        x_test = np.array(f[data_name][:,:,:n_features])


    # get prediction
    predicted_loss = student_model.predict(x_test,batch_size=batch_size)

    # load testing BSM samples
    with h5py.File(input_signal_file, 'r') as f:
        # only for Graph
        PID = np.array(f['ProcessID']) if 'ProcessID' in f.keys() else None
        all_bsm_data = f[data_name][:,:,:n_features] if PID is not None else None
        # test model on BSM data
        result_bsm = []
        for bsm_data_name, bsm_id in zip(BSM_SAMPLES, [33,30,31,32]):
            bsm_data = all_bsm_data[PID[:,0]==bsm_id] if PID is not None \
                else np.array(f[f'bsm_data_{bsm_data_name}'][:,:,:n_features])
            predicted_bsm_data = student_model.predict(bsm_data,batch_size=batch_size)
            result_bsm.append([bsm_data_name, predicted_bsm_data])

    # save results
    with h5py.File(output_result, 'w') as h5f:
        if history: h5f.create_dataset('loss', data=history.history['loss'])
        if history: h5f.create_dataset('val_loss', data=history.history['val_loss'])
        h5f.create_dataset('predicted_loss', data=predicted_loss)
        for bsm in result_bsm:
            h5f.create_dataset(f'predicted_loss_{bsm[0]}', data=bsm[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-train-file', type=str, help='Evaluated Teacher on train set')
    parser.add_argument('--input-test-file', type=str, help='Evaluated Teacher on test set')
    parser.add_argument('--input-signal-file', type=str, help='Evaluated Teacher on signals set')
    parser.add_argument('--data-name', type=str, help='Name of the data in the input h5')
    parser.add_argument('--n-features', type=int, default=3, help='First N features to train on')
    parser.add_argument('--teacher-loss_name', type=str, default='teacher_loss', help='Name of the loss dataset in the h5')
    parser.add_argument('--output-model-h5', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-model-json', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-history', type=str, help='Output file with the model training history', default='output/student_history.pickle')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--n-epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--distillation-loss', type=str, default='mse', help='Loss to use for distillation')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=3E-3, help='Learning rate')
    parser.add_argument('--node-size', default=32, type=int, help='To use smaller student model')
    parser.add_argument('--output-result', type=str, help='Output file with results', required=True)
    parser.add_argument('--output-dir', type=str, default='plots/')
    parser.add_argument('--particles-shuffle-strategy', type=str, default='none', help='How to shuffle particles : none / shuffle_all / shuffle_within_between_pid / shuffle_within_pid')
    parser.add_argument('--particles-shuffle-during', type=str, default='never', help='Shuffle particles during : never / train / predict / train_predict')

    args = parser.parse_args()
    knowledge_distillation(**vars(args))
