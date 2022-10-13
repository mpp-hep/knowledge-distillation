# Utility methods for the knowledge distillation code.

import os

import tensorflow as tf
from tensorflow import keras
from .universal_student import UniversalStudent
from .terminal_colors import tcols


def choose_student(student_type: str, hyperparams: dict) -> keras.models.Model:
    """Select and instantiate a certain type of interaction network."""
    switcher = {
        "unistudent": lambda: UniversalStudent(**hyperparams),
    }

    model = switcher.get(student_type, lambda: None)()

    if model is None:
        raise TypeError("Given interaction network model type is not implemented!")

    return model


def choose_loss(choice: str, from_logits: bool = True) -> keras.losses.Loss:
    """Construct a keras optimiser object with a certain learning rate given a string
    for the name of that optimiser.
    """

    switcher = {
        "categorical_crossentropy": lambda: keras.losses.CategoricalCrossentropy(
            from_logits=from_logits
        ),
        "kl_divergence": lambda: keras.losses.KLDivergence(),
        "softmax_with_crossentropy": lambda: tf.nn.softmax_cross_entropy_with_logits,
    }

    loss = switcher.get(choice, lambda: None)()
    if loss is None:
        raise TypeError("The given loss name is not implemented in the wrapper yet!")

    return loss


def print_training_attributes(train_hp: dict, model):
    """Prints model attributes so all interesting infromation is printed."""
    print("\nTraining Parameters")
    print("-------------------")
    print(tcols.OKGREEN + "Optimiser: \t\t" + tcols.ENDC, model.optimizer.get_config())
    print(tcols.OKGREEN + "Batch size: \t\t" + tcols.ENDC, train_hp["batch"])
    print(tcols.OKGREEN + "Training epochs:\t" + tcols.ENDC, train_hp["epochs"])
    print(tcols.OKGREEN + "Loss student: \t\t" + tcols.ENDC, model.student_loss_fn)
    print(tcols.OKGREEN + "Loss distill: \t\t" + tcols.ENDC, model.distillation_loss_fn)
    print("")
