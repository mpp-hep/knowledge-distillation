import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
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

def student(image_shape):
    quantize = False
    inp = Input((image_shape[1:]))
    x = Flatten()(inp)
    x = BatchNormalization()(x)
    x = QDense(64,
        kernel_quantizer="quantized_bits(6,0,0,alpha=1)",
        bias_quantizer="quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(64)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if quantize else \
        Activation('relu')(x)
    x = QDense(32,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(32)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if quantize else \
        Activation('relu')(x)
    x = QDense(32,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(32)(x)
    x = BatchNormalization()(x)
    x = QActivation("quantized_relu(6,0)")(x) \
        if quantize else \
        Activation('relu')(x)
    out = QDense(1,
        kernel_quantizer = "quantized_bits(6,0,0,alpha=1)",
        bias_quantizer = "quantized_bits(6,0,0,alpha=1)")(x) \
        if quantize else \
        Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.summary()
    # compile AE
    model.compile(optimizer=Adam(lr=3E-3, amsgrad=True),
        loss='mse')
    return model
