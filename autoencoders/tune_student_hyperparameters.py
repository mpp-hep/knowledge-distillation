import os
import h5py
import logging
import numpy as np
import keras_tuner
import tensorflow as tf
import argparse
import setGPU
import pickle

from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperParameters
from tensorboard import program
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
    )

import os
 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu

tracking_address = '../output/tb_logs' # the path of your log file for TensorBoard

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class HyperStudent(keras_tuner.HyperModel):

    def __init__(self, input_shape, distillation_loss, param_threshold=(4500,5000),quantized = True):#  4500,5000
        self.input_shape = input_shape
        self.distillation_loss = distillation_loss
        self.quantized = quantized
        self.num_layers = [2,3,4]
        self.num_params = [4,8,16,32, 64]
        self.quant_bits = [[4,1],[8,3],[16,6]]
        self.quant_idx = [] 
        # assign an index for each type of quantization, making sure that the chosen number is not present in any of the combinations
        flat_quant_bits = [x for xs in self.quant_bits for x in xs]
        for i,j in enumerate(self.quant_bits,start=1):
            while i in flat_quant_bits or i in self.quant_idx:
                i += 1
            self.quant_idx.append(i)


        model_configurations = []
        self.bits_configurations = []
        self.model_configurations = []
        self.num_conf_perNlayer = []
        self.idx_confs_perNlayer = [0]

        for nl in self.num_layers:
            grid_choices = np.tile(self.num_params, (nl,1))
            configs = np.array(np.meshgrid(*grid_choices)).T.reshape(-1, nl)

            model_configurations.append(configs.tolist())

        for nl in self.num_layers:
            # Creating all possible quantization configurations given the number of layers and bitwidths
            grid_choices = np.tile(self.quant_idx, (nl,1))
            configs = np.array(np.meshgrid(*grid_choices)).T.reshape(-1, nl,1)
            for i,j in zip(self.quant_idx,self.quant_bits):
                configs = np.where(configs == i, j, configs)
            self.bits_configurations.append(configs.tolist())

        model_configurations = [num for sublist in model_configurations for num in sublist]

        lastconflen = 0
        
        for config in model_configurations:
            params = self.compute_model_params(config)
            if params <= param_threshold[1] and params >= param_threshold[0]:
                self.model_configurations.append(config)
                if len(config) != lastconflen and lastconflen != 0:
                    # Counting the number of configurations for each number of layers (it is very likely that there is a better way to do it)
                    self.idx_confs_perNlayer.append(len(self.model_configurations) - 1) 
                lastconflen = len(config)
        self.idx_confs_perNlayer.append(len(self.model_configurations))
        self.num_conf_perNlayer = [j-i for i, j in zip(self.idx_confs_perNlayer[:-1], self.idx_confs_perNlayer[1:])]
    
        print('Total feasible configurations: ', len(self.model_configurations))

    def compute_model_params(self, config):
        total_params = 0
        total_params += np.prod(self.input_shape[1:])*config[0]
        total_params += config[-1]
        for i in range(len(config)-1):
            total_params += config[i]*config[i+1]
        return total_params 


    def build(self, hp):

        #quantized = True
        inputs = keras.Input(shape=self.input_shape[1:])
        x = layers.Flatten()(inputs)

        config_index = hp.Int("config_indx", min_value=0, max_value=len(self.model_configurations)-1, step=1)
        bits_index = 0
        selected_bits_conf = []
        if self.quantized:
            temp_idx = []
            lastrange = 0
            for i in self.idx_confs_perNlayer[1:]:
            # Selecting the set of configurations associated to the # of layers of config_index and setting up the index of the quantization configurations accordingly
                #selected_bits_conf = [num for num in self.bits_configurations if len(self.bits_configurations[self.bits_configurations.index(num)][0]) == len(self.model_configurations[i-1])][0]
                bits_index = hp.Int("bits_indx_" + str(len(self.model_configurations[i-1])), min_value=0, max_value=(len(self.quant_bits)**(len(self.model_configurations[i-1])) - 1), step=1)
                temp_idx.append(bits_index)
                temp_idx.append(len(self.model_configurations[i-1]))
                lastrange = i        
            lastrange = 0

            # Selecting appropriate bit index considering the number of layers in the model configuration selected
            bits_index = temp_idx[::2][temp_idx[1::2].index(next(idx for idx in temp_idx[1::2] if idx == len(self.model_configurations[config_index])))]
            selected_bits_conf = [num for num in self.bits_configurations if len(self.bits_configurations[self.bits_configurations.index(num)][0]) == len(self.model_configurations[config_index])][0]
            print("Bitwidth index selected: ", bits_index)
           

        i = 0
        # Number of hidden layers of the MLP is a hyperparameter.
        
        for units in self.model_configurations[config_index]:
            # Number of units of each layer are
            # different hyperparameters with different names.
            
            if self.quantized:

                print("Bitwidth of QDense layer # ",i)
                print(selected_bits_conf[bits_index][i])
                x = QDense(units=units, activation=quantized_relu(selected_bits_conf[bits_index][i][0],selected_bits_conf[bits_index][i][1]), \
                        kernel_quantizer=quantized_bits(selected_bits_conf[bits_index][i][0],selected_bits_conf[bits_index][i][1],alpha=1), \
                        bias_quantizer=quantized_bits(selected_bits_conf[bits_index][i][0],selected_bits_conf[bits_index][i][1],alpha=1), kernel_initializer='random_normal')(x)
                i += 1
            else:
                x = layers.Dense(units=units,activation='relu')(x)
            
        
        # The last layer contains 1 unit, which
        # represents the learned loss value
        
        
        if self.quantized:
            final_quant_idx = hp.Int("final_quant_idx", min_value=0, max_value = len(self.quant_bits)-1, step=1)
            final_quant = self.quant_bits[final_quant_idx]
            print("Bitwidth of final QDense layer")
            print(final_quant)
            outputs = QDense(1,kernel_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1),bias_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1))(x)
            outputs = QActivation(activation=quantized_relu(final_quant[0],final_quant[1]))(outputs)
        else:
            outputs = layers.Dense(units=1, activation='relu')(x)

        hyper_student = keras.Model(inputs=inputs, outputs=outputs)

        hyper_student.compile(
            optimizer=Adam(lr=3E-3, amsgrad=True),
            loss=self.distillation_loss
            )
        hyper_student.summary()
        return hyper_student


class FixedArchHyperStudent(keras_tuner.HyperModel):

    def __init__(self, input_shape, distillation_loss, n_nodes=[4,8,16]):
        self.input_shape = input_shape
        self.distillation_loss = distillation_loss
        self.quant_bits = [[4,1],[8,3],[16,6]]
        self.quant_idx = [] 
        self.n_nodes = n_nodes
        # assign an index for each type of quantization, making sure that the chosen number is not present in any of the combinations
        flat_quant_bits = [x for xs in self.quant_bits for x in xs]
        for i,j in enumerate(self.quant_bits,start=1):
            while i in flat_quant_bits or i in self.quant_idx:
                i += 1
            self.quant_idx.append(i)


        model_configurations = []
        self.bits_configurations = []
        self.model_configurations = []
        self.num_conf_perNlayer = []
        self.idx_confs_perNlayer = [0]


        # Creating all possible quantization configurations given the number of layers and bitwidths
        grid_choices = np.tile(self.quant_idx, (len(self.n_nodes),1))
        configs = np.array(np.meshgrid(*grid_choices)).T.reshape(-1, len(self.n_nodes),1)
        for i,j in zip(self.quant_idx,self.quant_bits):
            configs = np.where(configs == i, j, configs)
        self.bits_configurations = configs.tolist()


    def compute_model_params(self, config):
        total_params = 0
        total_params += np.prod(self.input_shape[1:])*config[0]
        total_params += config[-1]
        for i in range(len(config)-1):
            total_params += config[i]*config[i+1]
        return total_params 


    def build(self, hp):


        inputs = keras.Input(shape=self.input_shape[1:])
        x = layers.Flatten()(inputs)

        bits_index = hp.Int("bits_indx", min_value=0, max_value=len(self.bits_configurations)-1, step=1)
        print("Bitwidth index selected: ", bits_index)
           

        i = 0
        # Number of hidden layers of the MLP is a hyperparameter.
        
        for units in self.bits_configurations[bits_index]:
            # Number of units of each layer are
            # different hyperparameters with different names.
            
            print("Bitwidth of QDense layer # ",i)
            print(units)
            x = QDense(units=self.n_nodes[i], activation=quantized_relu(units[0],units[1]), \
                    kernel_quantizer=quantized_bits(units[0],units[1],alpha=1), \
                    bias_quantizer=quantized_bits(units[0],units[1],alpha=1), kernel_initializer='random_normal')(x)
            i += 1
            
        
        # The last layer contains 1 unit, which
        # represents the learned loss value
        
        
        final_quant_idx = hp.Int("final_quant_idx", min_value=0, max_value = len(self.quant_bits)-1, step=1)
        final_quant = self.quant_bits[final_quant_idx]
        print("Bitwidth of final QDense layer")
        print(final_quant)
        outputs = QDense(1,kernel_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1),bias_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1))(x)
        outputs = QActivation(activation=quantized_relu(final_quant[0],final_quant[1]))(outputs)
        

        hyper_student = keras.Model(inputs=inputs, outputs=outputs)

        hyper_student.compile(
            optimizer=Adam(lr=3E-3, amsgrad=True),
            loss=self.distillation_loss
            )
        hyper_student.summary()
        return hyper_student

def optimisation(input_file, distillation_loss, n_nodes, printconflut):
    printconflut = True
    # load teacher's loss for training
    with h5py.File(input_file, 'r') as f:
        x_train = np.array(f['data'][:,:,:3])
        y_train = np.array(f['teacher_loss'])

    if n_nodes:
        hypermodel = FixedArchHyperStudent(x_train.shape, distillation_loss, n_nodes)
        tuner = keras_tuner.RandomSearch(
              hypermodel,
              objective='val_loss',
              max_trials=len(hypermodel.bits_configurations),
              overwrite=True,
              directory='output/hyper_tuning_quant',
              )
    else:
        hypermodel = HyperStudent(x_train.shape, distillation_loss)
        #tuner = keras_tuner.Hyperband(
        #      hypermodel,
        #      objective='val_loss',
        #      #max_trials=len(hypermodel.model_configurations),
        #      overwrite=True,
        #      directory='output/hyper_tuning',
        #      max_epochs=100
        #      )
        if (printconflut == True):
            with open("model_confs",'wb') as f:
                # for i in hypermodel.model_configurations:
                #     # for j in i:
                #     #     f.write(str(j))
                #     #     f.write(',')
                #     f.write(str(i))
                #     f.write('\n')
                pickle.dump(hypermodel.model_configurations,f)

            with open("bits_confs",'wb') as f:
            #     for i in hypermodel.bits_configurations:
            #         f.write(str(i))
            #         f.write('\n')
                pickle.dump(hypermodel.bits_configurations,f)
            print('Index LUTs written on file')
            return 0
        tuner = keras_tuner.RandomSearch(
              hypermodel,
              objective='val_loss',
              max_trials=len(hypermodel.model_configurations),
              overwrite=True,
              directory='output/hyper_tuning',
              )
    tuner.search_space_summary()
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[
        TensorBoard(log_dir=tracking_address, histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]
    tuner.search(
        x=x_train,
        y=y_train,
        epochs=3,
        batch_size=2048,
        validation_split=0.2,
        callbacks=callbacks
        )

    tuner.results_summary()
    logging.info('Get the optimal hyperparameters')

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    logging.info('Getting and printing best hyperparameters!')
    if n_nodes:
        print('Optimal Configuration:', hypermodel.bits_configurations[best_hps['config_indx']])
    else:
        print('Optimal Configuration:', hypermodel.model_configurations[best_hps['config_indx']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--distillation-loss', type=str, default='mse', help='Loss to use for distillation')
    parser.add_argument('--n-nodes', nargs='+', type=int, default=None, help='# nodes for each layer for a search of optimal quantization with fixed network architecture')
    parser.add_argument('--printconflut', type=bool, default=False)
    args = parser.parse_args()
    optimisation(**vars(args))