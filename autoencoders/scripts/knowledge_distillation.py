import os
import argparse
import h5py
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )

from models import student
from plot_results import BSM_SAMPLES

import setGPU


def main(args):

    # load teacher's loss for training
    datasets = np.load(args.input_file)
    x_train = datasets['x_train'][:,:,:args.n_features]
    y_train = datasets['teacher_train_loss']
    x_test = datasets['x_test'][:,:,:args.n_features]
    y_test = datasets['teacher_test_loss']
    x_val = datasets['x_val'][:,:,:args.n_features]
    y_val = datasets['teacher_val_loss']

    # student model
    student_model = student(
        x_train.shape,
        args.learning_rate,
        args.dropout,
        args.node_size,
        args.distillation_loss,
        args.quant_size,
        args.particles_shuffle_strategy,
        args.particles_shuffle_during
        )

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]

    # train student to reproduce teachers' loss
    print('Starting training')
    history = student_model.fit(x=x_train, y=y_train,
        epochs=args.n_epochs,
        batch_size=args.batch_size,
        verbose=2,
        validation_data=(x_val,y_val),
        callbacks=callbacks)

    plt.hist(student_model.predict(x_train, batch_size=args.batch_size),
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
    plt.savefig(os.path.join(args.output_dir, f'loss_on_training.pdf'))

    # save student model
    #TO DO: With custom model it is not trivial to save model as a json. Either fix or remove save-json option
    #student_model_json = student_model.to_json()
    with open(args.output_model_json, 'w') as json_file:
        #json_file.write(student_model_json)
        json_file.write(json.dumps({}))
    student_model.save_weights(args.output_model_h5)

    # save training history
    if history:
        np.save(args.output_history, history.history)

    # get prediction
    predicted_loss = student_model.predict(x_test, batch_size=args.batch_size)

        signal_dict[f'bsm_data_{bsm_data_name}'] = bsm_data
        signal_dict[f'teacher_loss_{bsm_data_name}'] = data=np.log(bsm_loss+1) if args.log_loss else bsm_loss

    # load testing BSM samples
    signal_dict = np.load(args.input_signal_file)
    result_bsm = dict()
    for bsm_data_name, bsm_data in zip(signal_dict):
        if 'bsm_data_' in bsm_data_name:
            bsm_data = bsm_data[:,:,:args.n_features]
            predicted_bsm_data = student_model.predict(bsm_data, batch_size=args.batch_size)
            result_bsm[bsm_data_name] = predicted_bsm_data

    # save results
    np.savez(args.output_result,
        loss=history.history['loss'],
        val_loss=history.history['val_loss'],
        predicted_loss=predicted_loss,
        **result_bsm)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', type=str,
        help='Evaluated Teacher on train set')
    parser.add_argument('input_signal_file', type=str,
        help='Evaluated Teacher on signals set')

    parser.add_argument('--n-features', type=int, default=3,
        help='First N features to train on')
    parser.add_argument('--output-model-h5', type=str,
        help='Output file with the model', required=True)
    parser.add_argument('--output-model-json', type=str,
        help='Output file with the model', required=True)
    parser.add_argument('--output-history', type=str,
        help='Output file with the model training history', default='output/student_history.pickle')
    parser.add_argument('--batch-size', type=int, required=True,
        help='Batch size')
    parser.add_argument('--n-epochs', type=int, required=True,
        help='Number of epochs')
    parser.add_argument('--distillation-loss', type=str, default='mse',
        help='Loss to use for distillation')
    parser.add_argument('--dropout', type=float, default=None,
        help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=3E-3,
        help='Learning rate')
    parser.add_argument('--node-size', default=32, type=int,
        help='To use smaller student model')
    parser.add_argument('--quant-size', default=0, type=int,
        help='How much bits to use for quantization; 0 means full precision :D')
    parser.add_argument('--output-result', type=str,
        help='Output file with results', required=True)
    parser.add_argument('--output-dir', type=str, default='plots/')
    parser.add_argument('--particles-shuffle-strategy', type=str, default='none',
        choices=['none','shuffle_all', 'shuffle_within_between_pid','shuffle_within_pid'],
        help='How to shuffle particles : none / shuffle_all / shuffle_within_between_pid / shuffle_within_pid')
    parser.add_argument('--particles-shuffle-during', type=str, default='never',
        choices=['never','train','predict','train_predict'],
        help='Shuffle particles during : never / train / predict / train_predict')

    args = parser.parse_args()
    main(args)
