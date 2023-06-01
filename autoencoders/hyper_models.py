import os
import h5py
import numpy as np
import keras_tuner
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from keras_tuner import HyperParameters
from tensorboard import program
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
    )

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import math
from qkeras.utils import load_qmodel
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qnormalization import QBatchNormalization

tf.random.set_seed(1234)

tracking_address = '../output/tb_logs' # the path of your log file for TensorBoard

colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']

BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']

class HyperStudent(keras_tuner.HyperModel):

    def __init__(self, input_shape, distillation_loss, param_threshold=(5000,6000),quantized = True,expose_latent=False):#  4500,8000
        self.input_shape = input_shape
        self.distillation_loss = distillation_loss
        self.quantized = quantized
        self.expose_latent = expose_latent
        self.num_layers = [3,4,5,6]
        self.num_params = [8,16,32, 64]
        self.quant_bits = [[8,2],[16,6]]
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
            test = self.vivado_compatibility_check(config)
            if test == False:
                continue
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
    
        self.input_test_file = "output/l1_ae_test_loss.h5"
        self.input_signal_file = "output/l1_ae_signal_loss.h5"
        self.batch_size = 2048
        data_name = 'data'
        n_features = 3
        with h5py.File(self.input_test_file, 'r') as f:
            self.x_test = np.array(f[data_name][:,:,:n_features])
            self.teacher_total_loss = []
            self.teacher_total_loss.append(np.array(f['teacher_loss'])[:].flatten())
        with h5py.File(self.input_signal_file, 'r') as f:
            # only for Graph
            PID = np.array(f['ProcessID']) if 'ProcessID' in f.keys() else None
            all_bsm_data = f[data_name][:,:,:n_features] if PID is not None else None
            print("PID", PID)
            # test model on BSM data
            self.bsm_data = []
            self.teacher_signal_loss = {}
            for bsm_data_name, bsm_id in zip(BSM_SAMPLES, [33,30,31,32]):
                if PID is not None:
                    self.bsm_data.append(all_bsm_data[PID[:,0]==bsm_id])
                else:
                    self.bsm_data.append(np.array(f[f'bsm_data_{bsm_data_name}'][:,:,:n_features]))
                    self.teacher_signal_loss[bsm_data_name] = f[f'teacher_loss_{bsm_data_name}'][:].flatten()

        

    def compute_model_params(self, config):
        total_params = 0
        total_params += np.prod(self.input_shape[1:])*config[0]
        total_params += config[-1]
        for i in range(len(config)-1):
            total_params += config[i]*config[i+1]
        return total_params 
    def vivado_compatibility_check(self, config):
        dim = np.prod(self.input_shape[1:])*config[0]
        if dim > 1024:
            return False
        for i in range(len(config)-1):
            dim = config[i]*config[i+1]
            if dim > 1024:
                return False
        return True

    def build(self, hp):

        #quantized = True
        inputs = keras.Input(shape=self.input_shape[1:])
        x = layers.Flatten()(inputs)
        if self.quantized:
            x = QBatchNormalization(beta_quantizer='quantized_bits(8,6)', gamma_quantizer='quantized_bits(8,6)', mean_quantizer='quantized_bits(8,6)', variance_quantizer='quantized_bits(8,6)')(x)
        else:
            x = layers.BatchNormalization()(x)
        # print(self.model_configurations)
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
        if self.expose_latent == True: 
            latent_position = math.ceil(len(self.model_configurations[config_index])/2)
        else:
            latent_position = 0
        for units in self.model_configurations[config_index]:
            # Number of units of each layer are
            # different hyperparameters with different names.
            
            if self.quantized:

                print("Bitwidth of QDense layer # ",i)
                #print(selected_bits_conf)
                print(selected_bits_conf[bits_index][i])
                x = QDense(units=units, \
                        kernel_quantizer=quantized_bits(selected_bits_conf[bits_index][i][0],selected_bits_conf[bits_index][i][1],alpha=1), \
                        bias_quantizer=quantized_bits(selected_bits_conf[bits_index][i][0],selected_bits_conf[bits_index][i][1],alpha=1), kernel_initializer='he_normal', kernel_regularizer=L2(0.0001), name = f'qdense_{i}')(x)
                x = QActivation(activation=quantized_relu(selected_bits_conf[bits_index][i][0],selected_bits_conf[bits_index][i][1],negative_slope=0.25))(x)
                i += 1
            else:
                x = layers.Dense(units=units,activation='relu',kernel_initializer='random_normal')(x)

            if self.expose_latent == True and i == latent_position:
                if self.quantized:
                    latent = QDense(units=8, \
                            kernel_quantizer=quantized_bits(16,6,alpha=1), \
                            bias_quantizer=quantized_bits(16,6,alpha=1), kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(0.0001), name = f'qdense_latent')(x)
                    x = latent
                    x = QActivation(activation=quantized_relu(16,6,negative_slope=0.25))(x)
                    
                else:
                    latent = tf.keras.layers.Dense(units=8,kernel_initializer='random_normal')(x)
                    x = latent
                    x = tf.keras.activations.relu(x)
        
        # The last layer contains 1 unit, which
        # represents the learned loss value
        
        
        if self.quantized:
            final_quant_idx = hp.Int("final_quant_idx", min_value=0, max_value = len(self.quant_bits)-1, step=1)
            final_quant = self.quant_bits[final_quant_idx]
            print("Bitwidth of final QDense layer")
            print(final_quant)
            outputs = QDense(1,kernel_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1),bias_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1), kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=5,seed=1234),name=f'dense_fin')(x)
            outputs = QActivation(activation=quantized_relu(final_quant[0],final_quant[1],negative_slope=0.25))(outputs)
        else:
            outputs = layers.Dense(units=1, activation='relu')(x)

        if self.expose_latent == True:
            hyper_student = tf.keras.Model(inputs=inputs, outputs=(latent,outputs))
        else:
            hyper_student = keras.Model(inputs=inputs, outputs=outputs)

        hyper_student.compile(
            optimizer=Adam(lr=3E-3, amsgrad=True),
            loss=self.distillation_loss
            )
        hyper_student.summary()
        return hyper_student

    def get_config_from_hp(self, hp):

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
            final_quant_idx = hp.Int("final_quant_idx", min_value=0, max_value = len(self.quant_bits)-1, step=1)
            final_quant = self.quant_bits[final_quant_idx]
            return (self.model_configurations[config_index],selected_bits_conf[bits_index],final_quant)
        else:
            return [self.model_configurations[config_index]]


class FixedArchHyperStudent(keras_tuner.HyperModel):

    def __init__(self, input_shape, distillation_loss, n_nodes=[4,8,16]):
        self.input_shape = input_shape
        self.distillation_loss = distillation_loss
        self.quant_bits = [[8,3],[16,6]]
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

        self.input_test_file = "output/l1_ae_test_loss.h5"
        self.input_signal_file = "output/l1_ae_signal_loss.h5"
        self.batch_size = 2048
        data_name = 'data'
        n_features = 3
        with h5py.File(self.input_test_file, 'r') as f:
            self.x_test = np.array(f[data_name][:,:,:n_features])
            self.teacher_total_loss = []
            self.teacher_total_loss.append(np.array(f['teacher_loss'])[:].flatten())
        with h5py.File(self.input_signal_file, 'r') as f:
            # only for Graph
            PID = np.array(f['ProcessID']) if 'ProcessID' in f.keys() else None
            all_bsm_data = f[data_name][:,:,:n_features] if PID is not None else None
            print("PID", PID)
            # test model on BSM data
            self.bsm_data = []
            self.teacher_signal_loss = {}
            for bsm_data_name, bsm_id in zip(BSM_SAMPLES, [33,30,31,32]):
                if PID is not None:
                    self.bsm_data.append(all_bsm_data[PID[:,0]==bsm_id])
                else:
                    self.bsm_data.append(np.array(f[f'bsm_data_{bsm_data_name}'][:,:,:n_features]))
                    self.teacher_signal_loss[bsm_data_name] = f[f'teacher_loss_{bsm_data_name}'][:].flatten()


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
        x = QBatchNormalization()(x)
        bits_index = hp.Int("bits_indx", min_value=0, max_value=len(self.bits_configurations)-1, step=1)
        print("Bitwidth index selected: ", bits_index)
           

        i = 0
        # Number of hidden layers of the MLP is a hyperparameter.
        
        for units in self.bits_configurations[bits_index]:
            # Number of units of each layer are
            # different hyperparameters with different names.
            
            print("Bitwidth of QDense layer # ",i)
            print(units)

            x = QDense(units=self.n_nodes[i], \
                    kernel_quantizer=quantized_bits(units[0],units[1],alpha=1), \
                    bias_quantizer=quantized_bits(units[0],units[1],alpha=1), kernel_initializer='he_normal', kernel_regularizer=L2(0.0001), name = f'qdense_{i}')(x)
            x = QActivation(activation=quantized_relu(units[0],units[1],negative_slope=0.25))(x)
            i += 1
            
        
        # The last layer contains 1 unit, which
        # represents the learned loss value
        
        
        final_quant_idx = hp.Int("final_quant_idx", min_value=0, max_value = len(self.quant_bits)-1, step=1)
        final_quant = self.quant_bits[final_quant_idx]
        print("Bitwidth of final QDense layer")
        print(final_quant)
        outputs = QDense(1,kernel_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1),bias_quantizer=quantized_bits(final_quant[0],final_quant[1],alpha=1), kernel_initializer=keras.initializers.RandomUniform(minval=0, maxval=5,seed=1234),name=f'dense_fin')(x)
        outputs = QActivation(activation=quantized_relu(final_quant[0],final_quant[1],negative_slope=0.25))(outputs)
        

        hyper_student = keras.Model(inputs=inputs, outputs=outputs)

        hyper_student.compile(
            optimizer=Adam(lr=3E-3, amsgrad=True),
            loss=self.distillation_loss
            )
        hyper_student.summary()
        return hyper_student

    def get_quant_from_hp(self, hp):
        bits_index = hp.Int("bits_indx", min_value=0, max_value=len(self.bits_configurations)-1, step=1)
        final_quant_idx = hp.Int("final_quant_idx", min_value=0, max_value = len(self.quant_bits)-1, step=1)
        selected_bits_conf = self.bits_configurations[bits_index]
        final_quant = self.quant_bits[final_quant_idx]
        return (selected_bits_conf,final_quant)