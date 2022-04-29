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

from models import student
from plotting.plotting import reco_loss

def knowledge_distillation(teacher_input_h5, teacher_input_json,
    output_model_h5, output_model_json, output_history, batch_size, n_epochs,
    output_result):
    # magic trick to make sure that Lambda function works
    tf.compat.v1.disable_eager_execution()

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # load data
    with open('output/data_-1.pickle', 'rb') as f:
        x_train, y_train, x_test, y_test, all_bsm_data, pt_scaler = pickle.load(f)

    # load teacher model
    with open(teacher_input_json, 'r') as jsonfile:
        config = jsonfile.read()
    teacher_model = tf.keras.models.model_from_json(config,
        custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
            'QDense': QDense, 'QConv2D': QConv2D, 'QActivation': QActivation})
    teacher_model.load_weights(teacher_input_h5)
    teacher_model.summary()

    # student model
    image_shape = (x_train.shape[0], 19, 3, 1)
    student_model = student(image_shape)

    # define callbacks
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
        ]

    y_teacher = reco_loss(y_train, teacher_model.predict(x_train))
    # train
    history = student_model.fit(x=x_train, y=y_teacher,
        epochs=n_epochs,
        batch_size=batch_size,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks)

    # history = None


    # save student model
    student_model_json = student_model.to_json()
    with open(output_model_json, 'w') as json_file:
        json_file.write(student_model_json)
    student_model.save_weights(output_model_h5)

    # save training history
    if history:
        with open(output_history, 'wb') as handle:
            pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # get prediction
    predicted_loss = student_model.predict(x_test)

    # test model on BSM data
    result_bsm = []
    for i, bsm_data_name in enumerate(['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']):
        bsm_data = all_bsm_data[i]
        predicted_bsm_data = student_model.predict(bsm_data)
        result_bsm.append([bsm_data_name, predicted_bsm_data])

    #Save results
    with h5py.File(output_result, 'w') as h5f:
        if history: h5f.create_dataset('loss', data=history.history['loss'])
        if history: h5f.create_dataset('val_loss', data=history.history['val_loss'])
        h5f.create_dataset('QCD', data=y_test)
        h5f.create_dataset('predicted_loss', data=predicted_loss)
        for bsm in result_bsm:
            h5f.create_dataset(f'predicted_loss_{bsm[0]}', data=bsm[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-input-h5', type=str, help='Where is the model')
    parser.add_argument('--teacher-input-json', type=str, help='Where is the model')
    parser.add_argument('--output-model-h5', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-model-json', type=str, help='Output file with the model', required=True)
    parser.add_argument('--output-history', type=str, help='Output file with the model training history', default='output/student_history.pickle')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--n-epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--output-result', type=str, help='Output file with results', required=True)
    args = parser.parse_args()
    knowledge_distillation(**vars(args))