# Train the student network given a teacher.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(123)

import intnets.util
import intnets.plots
from intnets.data import Data
from . import util
from .distiller import Distiller
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx("float64")


def main(args):
    intnets.util.device_info()
    outdir = intnets.util.make_output_directory("trained_students", args["outdir"])
    intnets.util.save_hyperparameters_file(args, outdir)

    data_hp = args["data_hyperparams"]
    intnets.util.nice_print_dictionary("DATA DEETS", data_hp)
    jet_data = Data.shuffled(**data_hp, seed=args["seed"], jet_seed=args["jet_seed"])

    print("Importing the teacher network model...")
    teacher = keras.models.load_model(args["teacher"])

    print("Instantiating the student network model...")
    student = util.choose_student(args["student_type"], args["student"])

    print("Making the distiller...")
    args["distill"]["optimizer"] = intnets.util.choose_optimiser(
        args["distill"]["optimizer"], args["training_hyperparams"]["lr"]
    )
    distiller_hyperparams = args["distill"]
    distiller = Distiller(student, teacher)
    distiller.compile(**distiller_hyperparams)

    print(tcols.HEADER + "\nTEACHING THE STUDENT \U0001F4AA" + tcols.ENDC)
    print("====================")
    training_hyperparams = args["training_hyperparams"]
    util.print_training_attributes(training_hyperparams, distiller)
    history = distiller.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=training_hyperparams["epochs"],
        batch_size=training_hyperparams["batch"],
        verbose=2,
        callbacks=get_callbacks(),
        validation_split=0.3,
    )

    print(tcols.OKGREEN + "Saving student model to: " + tcols.ENDC, outdir)
    student.save(outdir, save_format="tf")
    plot_model_performance(history.history, outdir)


def plot_model_performance(history: dict, outdir: str):
    """Does different plots that show the performance of the trained model."""
    intnets.plots.loss_vs_epochs(
        outdir,
        history["student_loss"],
        history["val_student_loss"],
        "loss_epochs_student",
    )
    intnets.plots.accuracy_vs_epochs(
        outdir,
        history["acc"],
        history["val_acc"],
    )
    print(tcols.OKGREEN + "\nPlots done! " + tcols.ENDC)


def get_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=50)
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc", factor=0.8, patience=40
    )

    return [early_stopping, learning]
