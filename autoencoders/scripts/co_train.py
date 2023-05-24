import math
import argparse
import numpy as np
import pickle
import setGPU

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )

from models import student, teacher
from plot_results import BSM_SAMPLES

import setGPU

## just for testing things
config = {
    'learning_rate': 3e-3,
    'dropout': None,
    'node_size': 32,
    'distillation_loss': 'mae',
    'quant_size': 0, # means full precision is used
    'particles_shuffle_strategy': 'none',
    'particles_shuffle_during': 'never',
    'batch_size': 1024
}


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


@tf.function
def cov_loss(student_latent):
    mk = tf.math.reduce_mean(student_latent, keepdims = True, axis=0)
    sk = tf.math.reduce_std(student_latent, keepdims = True, axis=0)
    dk = (student_latent - mk)/sk
    da = tf.expand_dims(student_latent[:, 0:8], axis=-1)
    db = tf.expand_dims(student_latent[:, 8:], axis=-1)
    mat = tf.linalg.matmul(da, db, transpose_b=True)
    z = tf.math.reduce_mean(mat, axis=0)
    q = tf.math.reduce_mean(z**2)
    return q


@tf.function
def latent_loss(teacher_latent, student_latent):
    return mse(teacher_latent, student_latent) + cos(teacher_latent, student_latent) + 1


def main(args):
    # load the data
    with open(args.data_file, 'rb') as f:
        x_train, y_train, x_test, y_test, _, _, all_bsm_data, pt_scaler, _, _, _, _ = pickle.load(f)

    # student model
    student_model = student(
        x_train.shape,
        config['learning_rate'],
        config['dropout'],
        config['node_size'],
        config['distillation_loss'],
        config['quant_size'],
        config['particles_shuffle_strategy'],
        config['particles_shuffle_during'],
        expose_latent=True
        )

    # teacher model
    teacher_model = teacher(
        x_train.shape,
        3e-3,
        config['quant_size'],
        config['particles_shuffle_strategy'],
        config['particles_shuffle_during'],
        expose_latent=True
        )

    ## load and prepare the train and validation dataset:
    train_data, val_data, train_labels, val_labels = train_test_split(
        x_train, y_train, test_size=0.2, random_state=21
    )

    # build the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(buffer_size=1000).batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)

    # define the model optimizers
    teacher_optimizer = tf.keras.optimizers.Adam(config['learning_rate'], amsgrad=True)
    student_optimizer = tf.keras.optimizers.Adam(config['learning_rate'], amsgrad=True)

    latent_loss_factor = 0
    if args.include_latent_loss:
        latent_loss_factor = 1

    @tf.function
    def train_step(inputs):

        x = inputs[0]
        y = inputs[1]
        with tf.GradientTape() as t_tape, tf.GradientTape() as s_tape:
            tlatent, reconstruction = teacher_model(x, training=True)

            slatent, loss_prediction = student_model(x, training=True)

            t_loss, loss_signal  = teacher_loss(y, reconstruction)

            s_loss = student_loss(loss_signal, loss_prediction)

            l_loss = latent_loss(tlatent, slatent[-1, 0:8])

            total_loss = t_loss + s_loss*100 + latent_loss_factor*(l_loss + cov_loss(slatent)) #weight the student regularisation

        gradients_of_teacher = t_tape.gradient(total_loss, teacher_model.trainable_variables)
        gradients_of_student = s_tape.gradient(total_loss, student_model.trainable_variables)

        teacher_optimizer.apply_gradients(zip(gradients_of_teacher, teacher_model.trainable_variables))
        student_optimizer.apply_gradients(zip(gradients_of_student, student_model.trainable_variables))

        return t_loss, s_loss

    @tf.function
    def eval_step(inputs):
        x = inputs[0]
        y = inputs[1]
        tlatent, reconstruction = teacher_model(x, training=False)

        slatent, loss_prediction = student_model(x, training=False)

        t_loss, loss_signal  = teacher_loss(y, reconstruction)

        s_loss = student_loss(loss_signal, loss_prediction)

        l_loss = latent_loss(tlatent, slatent[-1, 0:8])

        return t_loss, s_loss

    ## begin the co-training proceadure:
    _callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]
    _callbacks_t = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]
    callbacks = tf.keras.callbacks.CallbackList(
        _callbacks, add_history=True, model=student_model)
    callbacks_t = tf.keras.callbacks.CallbackList(
        _callbacks_t, add_history=True, model=teacher_model)
    logs = {}
    logs_t = {}
    callbacks.on_train_begin(logs=logs)
    callbacks_t.on_train_begin(logs=logs_t)
    state_accumulator = []

    for epoch in range(30):
        print(f'starting train epoch ... {epoch}')
        # Training loop
        train_epoch_tloss_avg = tf.keras.metrics.Mean()
        train_epoch_sloss_avg = tf.keras.metrics.Mean()

        for i, (x, y) in enumerate(train_dataset):

            t_loss, s_loss = train_step((x,y))

            train_epoch_tloss_avg.update_state(t_loss)
            train_epoch_sloss_avg.update_state(s_loss)

            if i % 1000 == 0:
                print(f'{i/len(train_dataset)*100:.1f}% - {train_epoch_tloss_avg.result().numpy():.5} - {train_epoch_sloss_avg.result().numpy():.5}')

        print(f'starting val epoch ... {epoch}')

        val_epoch_tloss_avg = tf.keras.metrics.Mean()
        val_epoch_sloss_avg = tf.keras.metrics.Mean()

        for i, (x, y) in enumerate(val_dataset):

            t_loss, s_loss = eval_step((x, y))

            val_epoch_tloss_avg.update_state(t_loss)
            val_epoch_sloss_avg.update_state(s_loss)

            if i % 1000 == 0:
                print(f'{i/len(val_dataset)*100:.1f}% - {val_epoch_tloss_avg.result().numpy():.5} - {val_epoch_sloss_avg.result().numpy():.5}')

        state = (train_epoch_tloss_avg.result().numpy(),
                 train_epoch_sloss_avg.result().numpy(),
                 val_epoch_tloss_avg.result().numpy(),
                 val_epoch_sloss_avg.result().numpy())
        print(epoch, state)
        state_accumulator.append(state)

    #     logs['train_loss'].append(state[1])
        logs['train_loss'] = state[1]
        logs_t['train_loss'] = state[0]
        logs['val_loss'] = state[3]
        logs_t['val_loss'] = state[2]

        callbacks.on_epoch_end(epoch, logs=logs)
        callbacks_t.on_epoch_end(epoch, logs=logs_t)

    ### save the trained models:

    teacher_model.save_weights(f'{args.output_dir}teacher_ckpt/teacher_ckpt')
    student_model.save_weights(f'{args.output_dir}student_ckpt/student_ckpt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-file', type=str,
        help='Where is the data')
    parser.add_argument('--output-dir', type=str,
        help='Where to put the other generated data')
    parser.add_argument('--include-latent-loss', type=bool, default=False,
        help='include latent matching')

    args = parser.parse_args()
    main(args)