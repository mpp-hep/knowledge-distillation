import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Flatten,
    Activation,
    )
from qkeras import (
    QDense,
    QActivation,
    quantized_bits
    )
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

def kl_loss(y_true, y_pred):
    kl = y_true * (tf.math.log(y_true/y_pred))
    total_kl = tf.math.reduce_mean(kl)
    return total_kl

def smape(y_true, y_pred):
    return 100 * ( abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)) )

def student(image_shape, lr, dropout, node_size, distillation_loss):
    quantize = False
    inp = Input((image_shape[1:]))
    x = Flatten()(inp)
    x = BatchNormalization()(x)
    x = QDense(node_size*2,
        kernel_quantizer="quantized_bits(6,0,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(node_size*2)(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if quantize else \
        Activation('relu')(x)
    x = QDense(node_size,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(node_size)(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if quantize else \
        Activation('relu')(x)
    x = QDense(node_size,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(node_size)(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if quantize else \
        Activation('relu')(x)
    x = QDense(1,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(1)(x)
    out = QActivation("quantized_relu(6,0)")(x) \
        if quantize else \
        Activation('relu')(x)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    # compile AE
    model.compile(optimizer=Adam(lr=lr, amsgrad=True),
        loss=distillation_loss)
    return model
