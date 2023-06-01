import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, QActivation
import pickle
import setGPU
import json

import matplotlib.pyplot as plt


import hls4ml
import keras_tuner
import multiprocessing as mp

from models import student
from plot_results import BSM_SAMPLES, get_metric, get_threshold, LABELS, PLOTTING_LABELS
from tune_student_hyperparameters import HyperStudent, FixedArchHyperStudent,outer_compute_model_params

tf.random.set_seed(123)



def hlscomp(mod):
    print("INSIDE HLSCONV")
    config = hls4ml.utils.config_from_keras_model(mod, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(m, hls_config=config, output_dir='HLS_Project_' + str(mod.name),part='xc7z020clg400-1')
    hls_model.compile()
    return hls_model



def build_models(list_models,input_train_file, input_test_file, input_val_file, input_signal_file,data_name, n_features, teacher_loss_name, batch_size, n_epochs, distillation_loss,n_models, output_dir):
    
    with open(list_models, 'r') as fp:
        jsonlist = json.load(fp)

    # load teacher's loss for training
    with h5py.File(input_train_file, 'r') as f:
        x_train = np.array(f[data_name][:,:,:n_features])
        y_train = np.array(f[teacher_loss_name])

    # load teacher's loss for validation
    with h5py.File(input_val_file, 'r') as f:
        x_val = np.array(f[data_name][:,:,:n_features])
        y_val = np.array(f[teacher_loss_name])

    modelslist = []
    hpslist = []
    if jsonlist[0][0] == '0_place':
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

        with open("output/hpdumplocalnoquant128test.json", 'r') as fp:
            confslist = json.load(fp)


        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
            ]
        bestlist = []
        for i,hpsconf in enumerate(confslist[:n_models]):
            print(hpsconf)
            modelsconf=accepted_confs[hpsconf['config']['values']['config_indx']]
            print(modelsconf)
            hypermodel = FixedArchHyperStudent(x_train.shape,distillation_loss,modelsconf)
            print(jsonlist[i][1])
            hps = keras_tuner.engine.hyperparameters.deserialize(jsonlist[i][1])
            student_model = hypermodel.build(hps)
            print('Starting training')
            history = student_model.fit(x=x_train, y=y_train,epochs=n_epochs,batch_size=batch_size,verbose=1,validation_data=(x_val,y_val),callbacks=callbacks)
            # build_results(student_model,input_test_file,input_signal_file,batch_size,data_name,n_features)
            modelslist.append(build_results(student_model,input_test_file,input_signal_file,batch_size,data_name,n_features))

    else:
        hypermodel = HyperStudent(x_train.shape,distillation_loss,quantized=True)
        for i in jsonlist[:n_models]:
            hpslist.append(keras_tuner.engine.hyperparameters.deserialize(i))
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=1e-9)
            ]
        for j in hpslist:
            student_model = hypermodel.build(j)
            print('Starting training')
            history = student_model.fit(x=x_train, y=y_train,epochs=n_epochs,batch_size=batch_size,verbose=1,validation_data=(x_val,y_val),callbacks=callbacks)
            # build_results(student_model,input_test_file,input_signal_file,batch_size,data_name,n_features)
            modelslist.append(build_results(student_model,input_test_file,input_signal_file,batch_size,data_name,n_features))

    return modelslist
    # for i,m in enumerate(modelslist):
    #     # print("predictin model " + str(i))
    #     print("INSIDE HLSCONV" + str(i))
    #     config = hls4ml.utils.config_from_keras_model(m, granularity='name')
    #     print("AFTER CONFIG")
    #     hlsmodelslist.append(hls4ml.converters.convert_from_keras_model(m, hls_config=config, output_dir='HLS_Project' + str(i),part='xc7z020clg400-1'))
    #     print("AFTER CONVERTERS")
        
#def build_bsm_dataset():

def build_results(student_model,input_test_file='output/l1_ae_test_loss.h5',input_signal_file='output/l1_ae_signal_loss.h5',batch_size=1024,data_name="data",n_features=3):
    
    with h5py.File(input_test_file, 'r') as f:
        x_test = np.array(f[data_name][:,:,:n_features])
    predicted_loss = student_model.predict(x_test,batch_size=batch_size)
    

    with h5py.File(input_signal_file, 'r') as f:
        # only for Graph
        PID = np.array(f['ProcessID']) if 'ProcessID' in f.keys() else None
        all_bsm_data = f[data_name][:,:,:n_features] if PID is not None else None
        print("PID", PID)
        # test model on BSM data
        result_bsm = []
        for bsm_data_name, bsm_id in zip(BSM_SAMPLES, [33,30,31,32]):
            bsm_data = all_bsm_data[PID[:,0]==bsm_id] if PID is not None \
                else np.array(f[f'bsm_data_{bsm_data_name}'][:,:,:n_features])
            predicted_bsm_data = student_model.predict(bsm_data,batch_size=batch_size,verbose=1)
            # print(predicted_bsm_data)
            result_bsm.append([bsm_data_name, predicted_bsm_data])
    
    # with h5py.File(output_result, 'w') as h5f:,output_result="student_result-q.h5"
    #     h5f.create_dataset('predicted_loss', data=predicted_loss)
    #     for bsm in result_bsm:
    #         h5f.create_dataset(f'predicted_loss_{bsm[0]}', data=bsm[1])
    return predicted_loss, result_bsm

def plot_rocs(resultlist,anomaly=[],signal_file='output/l1_ae_signal_loss.h5',teacher_loss_name='teacher_loss',teacher_file='output/l1_ae_test_loss.h5'):

    print("Studying ", anomaly[0], "BSM event")
    tprscore = [anomaly[0]]
    
    with h5py.File(teacher_file, 'r') as data:
        teacher_total_loss = []
        teacher_total_loss.append(np.array(data[teacher_loss_name])[:].flatten())

    with h5py.File(signal_file, 'r') as bsm_data:
        # for graph anomalies loss are all in one array
        if 'ProcessID' in bsm_data.keys():
            teacher_total_loss.append(bsm_data[teacher_loss_name][bsm_data['ProcessID'][:,0]==anomaly[1]].flatten())
        else:
            teacher_total_loss.append(bsm_data[f'{teacher_loss_name}_{anomaly[0]}'][:].flatten())
    
    mse_teacher_tr = get_threshold(teacher_total_loss[0], 'mse ae')
    figure_of_merit = {
            'teacher': get_metric(teacher_total_loss[0], teacher_total_loss[1]),
            'student': 0
            }
    fprscore = 0
    tprscor1 = 0
    for i,rate in enumerate(figure_of_merit['teacher'][0]):
        if rate < 10**-5:
            continue
        else:
            tprscore.append(figure_of_merit['teacher'][1][i])
            tprscor1=figure_of_merit['teacher'][1][i]
            fprscore = rate
            break
    print("Teacher TRPSCORE")
    print(tprscor1)
    K = 0
    
    for student_predicted_loss,student_result_bsm in resultlist:
        # # load AE model
        # print(student_result_bsm)
        # print("////////")
        # print([s[1] for s in student_result_bsm if anomaly[0] in s[0]][0].flatten())
        student_total_loss = []
        student_total_loss.append(student_predicted_loss[:].flatten())
        student_total_loss.append([s[1] for s in student_result_bsm if anomaly[0] in s[0]][0].flatten())

        mse_student_tr = get_threshold(student_total_loss[0], 'mse ae')

        figure_of_merit['student'] = get_metric(student_total_loss[0], student_total_loss[1])

    #print(figure_of_merit['student'][0])
    # plt.rcParams["figure.figsize"] = (2,4)
        fprscore = 0
        tprscor1 = 0
        for i,rate in enumerate(figure_of_merit['student'][0]):
            if rate < 10**-5:
                continue
            else:
                tprscore.append(figure_of_merit['student'][1][i])
                tprscor1=figure_of_merit['student'][1][i]
                fprscore = rate
                break
        print("TPRSCORE ", K, "Student")
        print(tprscor1)

        if K >= 10 or tprscor1 >= 1.0:
            K+=1
            continue
        roc_student = plt.plot(figure_of_merit['student'][0], figure_of_merit['student'][1], "-",
            label=f'Cosearch student AUC = {figure_of_merit["student"][2]*100:.0f}%',
            linewidth=3, color=colors[K])
        K+=1


    roc_teacher = plt.plot(figure_of_merit['teacher'][0], figure_of_merit['teacher'][1], "-",
            label=f'teacher AUC = {figure_of_merit["teacher"][2]*100:.0f}%',
            linewidth=3, color='#016c59')

    plt.xlim(10**(-6),1)
    plt.ylim(10**(-6),1.2)
    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate', )
    plt.xlabel('False Positive Rate', )
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3)
    plt.vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
    plt.legend(loc='lower right', frameon=False, title=f'ROC {LABELS[anomaly[0]][0]}', )
    plt.tight_layout()

    return mse_student_tr, mse_teacher_tr,tprscore


# for bsm in zip(BSM_SAMPLES, [33,30,31,32]):
#         student_tr, teacher_tr = plot_rocs(args.student, args.teacher, args.signal,
#             args.teacher_loss_name, bsm)
#         plt.savefig(os.path.join(args.output_dir, f'student_rocs_{bsm[0]}.pdf'))
#         plt.clf()

if __name__ == '__main__':
    #mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    oldcolors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf','#6b33ff','#33ff33']
    colors = ['#CD9D5F','#C84761','#80454E','#B82BCA','#DEF7D9','#B47241','#DF0F75','#78A954','#0AF87C','#1BB986','#D5FD63','#816F25','#3B1FDC','#D4C9A9','#8737E4','#08238F','#39A2AD','#831221','#E56DDB','#951846','#DD6D9E','#AC2D61','#811D91','#16837D','#D362AD','#2797CE','#A78835','#3F5A8A','#1E8BB1','#E6C8DC','#3A2D05','#470FF4','#B1AC5C','#18C92D','#E68A7A','#8E5342','#93F0E2','#48BCD7','#9E28AC','#B86ABB','#831EBB','#58BD29','#4CCAB4','#FC3844','#8044F9','#5EEB7C','#9903FF','#7D99C3','#577090','#B69922']
    
    parser.add_argument('--list-models', type=str, help='List of models from hyperparameters search')
    parser.add_argument('--input-train-file', type=str, help='Evaluated Teacher on train set')
    parser.add_argument('--input-test-file', type=str, help='Evaluated Teacher on test set')
    parser.add_argument('--input-val-file', type=str, help='Evaluated Teacher on val set')
    parser.add_argument('--input-signal-file', type=str, help='Evaluated Teacher on signals set')
    parser.add_argument('--data-name', type=str, help='Name of the data in the input h5')
    parser.add_argument('--n-features', type=int, default=3, help='First N features to train on')
    parser.add_argument('--teacher-loss_name', type=str, default='teacher_loss', help='Name of the loss dataset in the h5')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size')
    parser.add_argument('--n-epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--distillation-loss', type=str, default='mse', help='Loss to use for distillation')
    # parser.add_argument('--best-student-list', type=str, help='Ordered list with students\' hyperparameters after tuning')
    parser.add_argument('--n-models', type=int, required=True, help='Actual # of models to build and analyze')
    parser.add_argument('--output-dir', type=str, default='output/tuning-analysis-plots/')

    args = parser.parse_args()
    #knowledge_distillation(**vars(args))
    #modlist = build_models(**vars(args))#args.list_models,args.input_train_file, args.input_test_file, args.input_val_file, args.input_signal_file,args.data_name, args.n_features, args.teacher_loss_name, args.batch_size, args.n_epochs, args.distillation_loss, args.n_models, args.output_dir)
    modlistres = build_models(**vars(args))
    print("Analyzing ", len(modlistres), " models")
    # results = []
    # for model in modlist:
    #     results.append(build_results(model,args.input_test_file,args.input_signal_file,args.batch_size,args.data_name,args.n_features))
    tottprscore = []
    for bsm in zip(BSM_SAMPLES, [33,30,31,32]):
        student_tr, teacher_tr,tprscore = plot_rocs(modlistres, bsm, args.input_signal_file, args.teacher_loss_name, args.input_test_file)
        tottprscore.append(tprscore)
        plt.savefig(os.path.join(args.output_dir, f'student_rocslocalpostquanttot_{bsm[0]}.pdf'))
        plt.clf()
    for anomaly in tottprscore:
        plt.plot([1e-5], [anomaly[1]], ".", color='#016c59')
        for K,point in enumerate(anomaly[2:]):
            plt.plot([1e-5], [point], ".", color=colors[K])
        plt.xlim(10**(-6),10**(-4))
        #plt.ylim(min(anomaly[1:])/2,max(anomaly[1:])*2)
        plt.semilogx()
        plt.semilogy()
        #plt.ylabel('True Positive Rate', )
        #plt.xlabel('False Positive Rate', )
        plt.vlines(1e-5, 0, 1.5, linestyles='--', color='#ef5675', linewidth=1)
        #plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir,f'zoomed_rocslocalpostquanttot_{anomaly[0]}.pdf'))
        #plt.show()
        plt.clf()
    np.array(tottprscore).tofile(os.path.join(args.output_dir,"testtprscorelocalpostquanttot.csv"),"\n")
