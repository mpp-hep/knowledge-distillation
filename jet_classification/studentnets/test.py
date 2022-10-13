# Test a chosen student model on test data and check its performance.
# Make plots that quantify the performance of the student.

import os
import json

import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(123)

import intnets.util
import intnets.plots
from intnets.data import Data
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx("float64")


def main(args):
    intnets.util.device_info()
    plots_dir = intnets.util.make_output_directory(
        args["model_dir"], f"plots_{args['seed']}"
    )

    hyperparams = intnets.util.load_hyperparameters_file(args["model_dir"])
    hyperparams["data_hyperparams"].update(args["data_hyperparams"])

    data_hp = hyperparams["data_hyperparams"]
    intnets.util.nice_print_dictionary("DATA DEETS", data_hp)
    jet_data = Data.shuffled(**data_hp, jet_seed=args["jet_seed"], seed=args["seed"])

    print(tcols.HEADER + "Importing the model..." + tcols.ENDC)
    intnets.util.nice_print_dictionary("", hyperparams["student"])
    model = keras.models.load_model(args["model_dir"])
    model.summary(expand_nested=True)
    print(tcols.OKGREEN + "Model loaded! \U0001F370\U00002728\n" + tcols.ENDC)

    y_pred = tf.nn.softmax(model.predict(jet_data.te_data)).numpy()
    y_pred.astype("float32").tofile(os.path.join(plots_dir, "y_pred.dat"))
    print(tcols.OKGREEN + "\nSaved predictions array.\n" + tcols.ENDC)

    intnets.plots.dnn_output(plots_dir, y_pred)
    intnets.plots.roc_curves(plots_dir, y_pred, jet_data.te_target)
    print(tcols.OKGREEN + "\nPlotting done! \U0001F4C8\U00002728" + tcols.ENDC)
