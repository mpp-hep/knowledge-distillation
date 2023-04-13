import os
import h5py
import logging
import numpy as np
import keras_tuner
import tensorflow as tf
import argparse
import setGPU
import pickle
import json
from tensorflow import keras
import matplotlib.pyplot as plt

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

import os
 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from plot_results import get_metric

from qkeras.utils import load_qmodel
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
from qkeras.qnormalization import QBatchNormalization
tf.random.set_seed(1234)
tracking_address = '../output/tb_logs' # the path of your log file for TensorBoard
colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']
BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class HyperStudent(keras_tuner.HyperModel):

    def __init__(self, input_shape, distillation_loss, param_threshold=(5000,6000),quantized = True):#  4500,8000
        self.input_shape = input_shape
        self.distillation_loss = distillation_loss
        self.quantized = quantized
        self.num_layers = [3,4,5]
        self.num_params = [8,16,32, 64]
        self.quant_bits = [[8,3],[16,6]]
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



    def fit(self, hp, model, x, y, **kwargs):
        model.fit(x, y, **kwargs)
        config = []
        for l in model.layers[3:-2]:
            if 'dense' not in l.name:
                continue
            config.append(l.output_shape[1])
        modelname = str([a for a in config]).replace(', ','_').replace('[','').replace(']','')
        if self.quantized:
            config = ['_Bits']
            for l in model.layers[3:-1]:
                if 'dense' not in l.name:
                    continue
                config.append('(')
                config.append(l.kernel_quantizer.bits)
                config.append(l.kernel_quantizer.integer)
                config.append(')')
            config = str(config).replace('\'','').replace('(, ','(').replace(', )',')').replace(', ','_').replace('[','').replace(']','') 
            modelname += config
        else:
            config = "_" + str(model.layers[-2].output.shape[1])
            modelname += config


        #modelname = str([a for a in config]).replace(', ','_').replace('[','').replace(']','').replace('\'','')
        print(modelname)
        predicted_loss = model.predict(self.x_test,batch_size=self.batch_size)
        score_bsm = 0
        if not os.path.isdir(f'output/search_results/new_batch/{modelname}'):
            os.makedirs(f'output/search_results/new_batch/{modelname}')
        for i,bsm_data_name in enumerate(BSM_SAMPLES):
            predicted_bsm_data = model.predict(self.bsm_data[i],batch_size=self.batch_size,verbose=1)
            result_bsm = [bsm_data_name, predicted_bsm_data]
            
            teacher_merit = get_metric(self.teacher_total_loss[0], self.teacher_signal_loss[bsm_data_name])

            student_total_loss = []
            student_total_loss.append(predicted_loss[:].flatten())
            student_total_loss.append(result_bsm[1].flatten())
            # mse_student_tr = get_threshold(student_total_loss[0], 'mse ae')
            figure_of_merit = get_metric(student_total_loss[0], student_total_loss[1])
            fprscore = 0
            tprscor1 = 0
            for i,rate in enumerate(figure_of_merit[0]):
                if rate < 10**-5:
                    #print(figure_of_merit[1][i])
                    continue
                else:
                    if (rate >= (1.5*10**-5)):
                        tprscor1 = 0
                        fprscore = rate
                        break
                    tprscor1=figure_of_merit[1][i]
                    fprscore = rate
                    break
            score_bsm += (tprscor1 + figure_of_merit[2])/2
            print("TPRSCORE Student", tprscor1)
            roc_student = plt.plot(figure_of_merit[0], figure_of_merit[1], "-",
                            label=f'Cosearch student AUC = {figure_of_merit[2]*100:.0f}%',
                            linewidth=3, color=colors[1])
            roc_teacher = plt.plot(teacher_merit[0], teacher_merit[1], "-",
                            label=f'teacher AUC = {teacher_merit[2]*100:.0f}%',
                            linewidth=3, color='#016c59')
            plt.xlim(10**(-6),1)
            plt.ylim(10**(-6),1.2)
            plt.semilogx()
            plt.semilogy()
            plt.ylabel('True Positive Rate', )
            plt.xlabel('False Positive Rate', )
            plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3)
            plt.vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
            plt.legend(loc='lower right', frameon=False, title=f'ROC {bsm_data_name}', )
            plt.tight_layout()

            plt.savefig(os.path.join(f'output/search_results/new_batch/{modelname}', f'student_{bsm_data_name}_{modelname}.pdf'))
            plt.clf()
        model.save(os.path.join(f'output/search_results/new_batch/{modelname}', f'student_{modelname}.h5'))
        score_bsm /= 4
        score_bsm = 1 - score_bsm
        if score_bsm == 0.25:
            return 5
        else:
            return score_bsm


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
        

    def fit(self, hp, model, x, y, **kwargs):
        model.fit(x, y, **kwargs)
        config = []
        for l in model.layers[3:-2]:
            if 'dense' not in l.name:
                continue
            config.append(l.output_shape[1])
        modelname = str([a for a in config]).replace(', ','_').replace('[','').replace(']','')
        config = ['_Bits']
        for l in model.layers[3:-1]:
            if 'dense' not in l.name:
                continue
            config.append('(')
            config.append(l.kernel_quantizer.bits)
            config.append(l.kernel_quantizer.integer)
            config.append(')')
        config = str(config).replace('\'','').replace('(, ','(').replace(', )',')').replace(', ','_').replace('[','').replace(']','') 
        modelname += config



        #modelname = str([a for a in config]).replace(', ','_').replace('[','').replace(']','').replace('\'','')
        print(modelname)
        predicted_loss = model.predict(self.x_test,batch_size=self.batch_size)
        score_bsm = 0
        if not os.path.isdir(f'output/search_results_post/{modelname}'):
            os.makedirs(f'output/search_results_post/{modelname}')
        for i,bsm_data_name in enumerate(BSM_SAMPLES):
            predicted_bsm_data = model.predict(self.bsm_data[i],batch_size=self.batch_size,verbose=1)
            result_bsm = [bsm_data_name, predicted_bsm_data]
            
            teacher_merit = get_metric(self.teacher_total_loss[0], self.teacher_signal_loss[bsm_data_name])

            student_total_loss = []
            student_total_loss.append(predicted_loss[:].flatten())
            student_total_loss.append(result_bsm[1].flatten())
            # mse_student_tr = get_threshold(student_total_loss[0], 'mse ae')
            figure_of_merit = get_metric(student_total_loss[0], student_total_loss[1])
            fprscore = 0
            tprscor1 = 0
            for i,rate in enumerate(figure_of_merit[0]):
                if rate < 10**-5:
                    #print(figure_of_merit[1][i])
                    continue
                else:
                    if (rate >= (1.5*10**-5)):
                        tprscor1 = 0
                        fprscore = rate
                        break
                    tprscor1=figure_of_merit[1][i]
                    fprscore = rate
                    break
            score_bsm += (tprscor1 + figure_of_merit[2])/2
            print("TPRSCORE Student", tprscor1)
            roc_student = plt.plot(figure_of_merit[0], figure_of_merit[1], "-",
                            label=f'Cosearch student AUC = {figure_of_merit[2]*100:.0f}%',
                            linewidth=3, color=colors[1])
            roc_teacher = plt.plot(teacher_merit[0], teacher_merit[1], "-",
                            label=f'teacher AUC = {teacher_merit[2]*100:.0f}%',
                            linewidth=3, color='#016c59')
            plt.xlim(10**(-6),1)
            plt.ylim(10**(-6),1.2)
            plt.semilogx()
            plt.semilogy()
            plt.ylabel('True Positive Rate', )
            plt.xlabel('False Positive Rate', )
            plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3)
            plt.vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
            plt.legend(loc='lower right', frameon=False, title=f'ROC {bsm_data_name}', )
            plt.tight_layout()

            plt.savefig(os.path.join(f'output/search_results_post/{modelname}', f'student_{bsm_data_name}_{modelname}.pdf'))
            plt.clf()
        model.save(os.path.join(f'output/search_results_post/{modelname}', f'student_{modelname}.h5'))
        score_bsm /= 4
        score_bsm = 1 - score_bsm
        if score_bsm == 0.25:
            return 5
        else:
            return score_bsm

def optimisation(input_file, distillation_loss, n_nodes, printconflut,model_to_quant):
    printconflut = False
    # load teacher's loss for training
    with h5py.File(input_file, 'r') as f:
        x_train = np.array(f['data'][:,:,:3])
        y_train = np.array(f['teacher_loss'])
    
    # load teacher's loss for validation
    with h5py.File("output/l1_ae_val_loss.h5", 'r') as f:
        x_val = np.array(f['data'][:,:,:3])
        y_val = np.array(f['teacher_loss'])

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
              # objective='val_loss',
              max_trials=len(hypermodel.model_configurations),
              overwrite=True,
              directory='output/hyper_tuning_2',
              )
    tuner.search_space_summary()
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[
        # TensorBoard(log_dir=tracking_address, histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=3, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
        ]
    tuner.search(
        x=x_train,
        y=y_train,
        epochs=4,
        batch_size=2048,
        validation_data=(x_val,y_val),
        callbacks=callbacks
        )

    tuner.results_summary()
    logging.info('Get the optimal hyperparameters')

    best_hps = tuner.get_best_hyperparameters(num_trials=len(hypermodel.model_configurations))

    logging.info('Getting and printing best hyperparameters!')
    if n_nodes:
        print('Optimal Configuration:', hypermodel.bits_configurations[best_hps['config_indx']])
    else:
        print('Optimal Configuration:', hypermodel.model_configurations[best_hps[0]['config_indx']])
    bestlist = []
    for i in best_hps:
        bestlist.append(keras_tuner.engine.hyperparameters.serialize(i))
    with open('output/COquantsearchhp5000_6000_newbatch.json', 'w') as fp:
        json.dump(bestlist, fp)


def outer_compute_model_params(input_shape,config):
        total_params = 0
        total_params += np.prod(input_shape)*config[0]
        total_params += config[-1]
        for i in range(len(config)-1):
            total_params += config[i]*config[i+1]
        return total_params

def multioptimisation(input_file, distillation_loss,modelslist):
    # load teacher's loss for training
    with h5py.File(input_file, 'r') as f:
        x_train = np.array(f['data'][:,:,:3])
        y_train = np.array(f['teacher_loss'])

    # load teacher's loss for validation
    with h5py.File("output/l1_ae_val_loss.h5", 'r') as f:
        x_val = np.array(f['data'][:,:,:3])
        y_val = np.array(f['teacher_loss'])

    # The logs will be write to "/tmp/tb_logs".
    callbacks=[
        #TensorBoard(log_dir=tracking_address, histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=2, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, min_lr=1e-9)
        ]

    num_layers = [2,3,4]
    num_params = [16,32, 64,128]
    model_configurations = []
    for nl in num_layers:
        grid_choices = np.tile(num_params, (nl,1))
        configs = np.array(np.meshgrid(*grid_choices)).T.reshape(-1, nl)
        model_configurations.append(configs.tolist())

    model_configurations = [num for sublist in model_configurations for num in sublist]
    accepted_confs = []
    for config in model_configurations:
        params = outer_compute_model_params([19,3,1],config)
        if params <= 8000 and params >= 4500:
            accepted_confs.append(config)

    with open(modelslist, 'r') as fp:
        jsonlist = json.load(fp)

    bestlist = []
    for i,hpsconf in enumerate(jsonlist[:10]):
        hypermodel = HyperStudent(x_train.shape,distillation_loss, param_threshold=(3000,6000), quantized=False)
        hps = keras_tuner.engine.hyperparameters.deserialize(hpsconf)
        layout = hypermodel.get_config_from_hp(hps)[0]
        del hypermodel
        #modelsconf=accepted_confs[hpsconf['config']['values']['config_indx']]
        print(layout)
        hypermodel = FixedArchHyperStudent(x_train.shape, distillation_loss, layout)
        tuner = keras_tuner.RandomSearch(
              hypermodel,
              max_trials=len(hypermodel.bits_configurations),
              overwrite=True,
              directory='output/hyper_tuning_quant_' + str(i),
              )
        tuner.search_space_summary()
        tuner.search(
            x=x_train,
            y=y_train,
            epochs=3,
            batch_size=4096,
            validation_data = (x_val,y_val),
            callbacks=callbacks
            )

        tuner.results_summary()
        logging.info('Get the optimal hyperparameters')

        best_hps = tuner.get_best_hyperparameters(num_trials=len(hypermodel.bits_configurations))
        print(best_hps)
        # finmod = tuner.hypermodel.build(best_hps)
        logging.info('Getting and printing best hyperparameters!')
        print('Optimal Configuration:', hypermodel.bits_configurations[best_hps[0]['bits_indx']], hypermodel.quant_bits[best_hps[0]['final_quant_idx']])

        bestquant = [str(i)+"_place"]
        for i in best_hps:
            bestquant.append(keras_tuner.engine.hyperparameters.serialize(i))
        bestlist.append(bestquant)
    with open('Postsearchhp3000_6000.json', 'w') as fp:
        json.dump(bestlist, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='input file', required=True)
    parser.add_argument('--distillation-loss', type=str, default='mse', help='Loss to use for distillation')
    parser.add_argument('--n-nodes', nargs='+', type=int, default=None, help='# nodes for each layer for a search of optimal quantization with fixed network architecture')
    parser.add_argument('--printconflut', type=bool, default=False)
    parser.add_argument('--model-to-quant', type=str, default=None, help='json produced by keras tuner with hyperparameters to perform QAT with fixed archs')
    args = parser.parse_args()
    if (args.model_to_quant):
        multioptimisation(args.input_file,args.distillation_loss,args.model_to_quant)
    else:
        optimisation(**vars(args))
