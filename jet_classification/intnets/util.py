# Utility methods for the interaction network training, testing, etc...

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras

from .terminal_colors import tcols
from .qconvintnet import QConvIntNet
from .convintnet import ConvIntNet
from .densintnet import DensIntNet


def make_output_directory(location: str, outdir: str) -> str:
    """Create the output directory in a designated location."""
    outdir = os.path.join(location, outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def nice_print_dictionary(dictionary_name: str, dictionary: dict):
    """Logs useful details about the data used to train the interaction network."""
    print(tcols.HEADER + f"\n{dictionary_name}" + tcols.ENDC)
    print(tcols.HEADER + "-----------" + tcols.ENDC)
    for key in dictionary:
        print(f"{key}: {dictionary[key]}")


def device_info():
    """Prints what device the tensorflow network will run on."""
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        print(tcols.OKCYAN + f"\nGPU: {details.get('device_name')}" + tcols.ENDC)
    else:
        print(tcols.WARNING + "\nNo GPU detected. Running on CPU." + tcols.ENDC)


def choose_optimiser(choice: str, lr: float) -> keras.optimizers.Optimizer:
    """Construct a keras optimiser object with a certain learning rate given a string
    for the name of that optimiser.
    """

    switcher = {
        "adam": lambda: keras.optimizers.Adam(learning_rate=lr),
    }

    optimiser = switcher.get(choice, lambda: None)()
    if optimiser is None:
        raise TypeError(
            "Optimiser was not implemented yet! Please choose one of the "
            "following: adam."
        )

    return optimiser


def print_training_attributes(model: keras.models.Model, args: dict):
    """Prints model attributes so all interesting infromation is printed."""
    compilation_hyperparams = args["intnet_compilation"]
    train_hyperparams = args["training_hyperparams"]

    print("\nTraining parameters")
    print("-------------------")
    print(tcols.OKGREEN + "Optimiser: \t" + tcols.ENDC, model.optimizer.get_config())
    print(tcols.OKGREEN + "Batch size: \t" + tcols.ENDC, train_hyperparams["batch"])
    print(tcols.OKGREEN + "Learning rate: \t" + tcols.ENDC, train_hyperparams["lr"])
    print(tcols.OKGREEN + "Training epochs:" + tcols.ENDC, train_hyperparams["epochs"])
    print(tcols.OKGREEN + "Loss: \t\t" + tcols.ENDC, compilation_hyperparams["loss"])
    print("")


def choose_intnet(
    intnet_type: str,
    nconst: int,
    nfeats: int,
    hyperparams: dict,
    compilation_hyperparams,
) -> keras.models.Model:
    """Select and instantiate a certain type of interaction network."""
    print("Instantiating model with the hyperparameters:")
    for key in hyperparams:
        print(f"{key}: {hyperparams[key]}")

    switcher = {
        "qconv": lambda: QConvIntNet(nconst, nfeats, **hyperparams),
        "dens": lambda: DensIntNet(nconst, nfeats, **hyperparams),
        "conv": lambda: ConvIntNet(nconst, nfeats, **hyperparams),
    }

    model = switcher.get(intnet_type, lambda: None)()

    compilation_hyperparams["optimizer"] = choose_optimiser(
        compilation_hyperparams["optimizer"][0],
        compilation_hyperparams["optimizer"][1],
    )
    compilation_hyperparams["loss"] = choose_loss(compilation_hyperparams["loss"])
    model.compile(**compilation_hyperparams)
    model.build((None, nconst, nfeats))
    print(tcols.OKGREEN + "Model compiled and built!" + tcols.ENDC)

    if intnet_type == "qconv":
        set_matrix_multiplication_hack_weights(model)

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
        "softmax_with_crossentropy": lambda: tf.nn.softmax_cross_entropy_with_logits,
    }

    loss = switcher.get(choice, lambda: None)()
    if loss is None:
        raise TypeError("The given loss name is not implemented in the wrapper yet!")

    return loss


def save_hyperparameters_file(hyperparams: dict, outdir: str):
    """Saves the hyperparameters dictionary that defines an net to a file."""
    hyperparams_file_path = os.path.join(outdir, "hyperparameters.json")
    with open(hyperparams_file_path, "w") as file:
        json.dump(hyperparams, file)

    print(tcols.OKGREEN + "Saved hyperparameters to json file." + tcols.ENDC)


def load_hyperparameters_file(model_dir: str):
    """Loads a hyperparameters file given the directory that it's in."""
    with open(os.path.join(model_dir, "hyperparameters.json")) as file:
        hyperparams = json.load(file)

    return hyperparams


def set_matrix_multiplication_hack_weights(model: keras.models.Model):
    """Set the weights of the QKeras convolutional layers that are used to do
    quantised matrix multiplication to be equivalent to the elements of the first
    matrix in the multiplication. This hack to do matrix multiplication is required
    since QKeras does not have yet an implementation of matrix multiplication.
    """
    transpose_receiver = np.transpose(model._receiver_matrix)
    model.get_layer("ORr").set_weights([np.expand_dims(model._receiver_matrix, axis=0)])
    model.get_layer("ORs").set_weights([np.expand_dims(model._sender_matrix, axis=0)])
    model.get_layer("Ebar").set_weights([np.expand_dims(transpose_receiver, axis=0)])
