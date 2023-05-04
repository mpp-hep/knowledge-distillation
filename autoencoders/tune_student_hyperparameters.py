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

from hyper_models import HyperStudent,FixedArchHyperStudent

tf.random.set_seed(1234)
tracking_address = '../output/tb_logs' # the path of your log file for TensorBoard
colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']
BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class HyperStudent(HyperStudent):
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


class FixedArchHyperStudent(FixedArchHyperStudent):
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
        hypermodel = HyperStudent(x_train.shape, distillation_loss, param_threshold=(5000,6000),quantized = True)
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
