import os
import argparse
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
    )
# import tensorflow_model_optimization as tfmot
# from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, QActivation
from qkeras.utils import load_qmodel
import pickle
import setGPU
import json
import math
import matplotlib.pyplot as plt

from plot_results import BSM_SAMPLES, get_metric, get_threshold, LABELS, PLOTTING_LABELS
from models import teacher

tf.random.set_seed(123)

def mse_loss(inputs, outputs):
    return tf.math.reduce_mean(tf.math.square(outputs-inputs), axis=-1)

def make_mse(inputs, outputs):
    # remove last dimension
    inputs = tf.squeeze(inputs, axis=-1)
    inputs = tf.cast(inputs, dtype=tf.float32)
    # trick with phi
    outputs_phi = math.pi*tf.math.tanh(outputs)
    # trick with phi
    outputs_eta_egamma = 3.0*tf.math.tanh(outputs)
    outputs_eta_muons = 2.1*tf.math.tanh(outputs)
    outputs_eta_jets = 4.0*tf.math.tanh(outputs)
    outputs_eta = tf.concat([outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    # use both tricks
    outputs = tf.concat([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # mask zero features
    mask = tf.math.not_equal(inputs,0)
    mask = tf.cast(mask, tf.float32)
    outputs = mask * outputs

    lossm = mse_loss(tf.reshape(inputs, [-1, 57]), tf.reshape(outputs, [-1, 57]))
    loss = tf.math.reduce_mean(lossm, axis=0) # average over batch
    return loss, lossm


@tf.function
def student_loss(true_loss, pred_loss):
    return mae(true_loss, pred_loss)

@tf.function
def teacher_loss(inputs, outputs):
    return make_mse(inputs, outputs)

def build_models(input_test_file, input_signal_file, models_dir, data_name, n_features, batch_size, output_dir):
    data_name="data"
    n_features=3
    result_dir = "output/cotrain_results"
    l = os.listdir(result_dir)
    mse = []
    teacher_loss_sgn = []
    all_bsm_data = 0
    bsm_sign = []
    with open("output/kd_output/data_-1.pickle", 'rb') as f:
        _, _, x_test, y_test, _, _, all_bsm_data, pt_scaler, _, _, _, _ = pickle.load(f)
    # with h5py.File(input_test_file, 'r') as f:
    #     x_test = np.array(f[data_name][:,:,:n_features])
    #with h5py.File(input_signal_file, 'r') as f:
    #    # only for Graph
    #    PID = np.array(f['ProcessID']) if 'ProcessID' in f.keys() else None
    #    all_bsm_data = f[data_name][:,:,:n_features] if PID is not None else None
    #    print("PID", PID)
    #    # test model on BSM data
    #    predicted_loss_sgn = []
    #    for bsm_data_name, bsm_id in zip(BSM_SAMPLES, [33,30,31,32]):
    #        bsm_data = all_bsm_data[PID[:,0]==bsm_id] if PID is not None \
    #            else np.array(f[f'bsm_data_{bsm_data_name}'][:,:,:n_features])
    #        bsm_sign.append(bsm_data)
    print(x_test.shape)
    for i,m in enumerate(l[:]):
        single_mse = []
        student_model = load_qmodel(result_dir + "/"+ m +"/student_"+ m + ".h5")
        teacher_model = teacher(
        x_test.shape,
        3e-3,
        0,
        'none',
        'never',
        expose_latent=True
        )
        _,_=teacher_model.predict(x_test,batch_size=4096)
        teacher_model.load_weights(result_dir + "/"+ m +"/teacher_model.h5")
        student_model.summary()
        print("Predicting model # ",i)
        _,reco_bkg = teacher_model.predict(x_test,batch_size=4096)
        _,predicted_loss = student_model.predict(x_test,batch_size=4096)
        print("SHAPES",y_test.shape,reco_bkg.shape)
        _, teacher_loss_bkg = teacher_loss(y_test, reco_bkg)
        teacher_loss_bkg = teacher_loss_bkg.numpy()
    # print(predicted_loss)
    # print(teacher_total_loss)
        # single_mse.append(mean_squared_error(teacher_loss_bkg, predicted_loss))
        fig, axs = plt.subplots(2,2)
        axs = axs.flatten()
        for j,s in enumerate(BSM_SAMPLES):
            bsm_data = all_bsm_data[j]
            _,predicted_loss_sgn = student_model.predict(bsm_data,batch_size=4096)
            _,reco_sgn = teacher_model.predict(bsm_data,batch_size=4096)
              
            bsm_data_target = np.copy(bsm_data)
            bsm_data_target = np.squeeze(bsm_data_target, axis=-1)
            bsm_data_target[:,:,0] = pt_scaler.transform(bsm_data_target[:,:,0])
            print("bsm shapes",bsm_data_target.shape,bsm_data_target[:,:,0].shape)
            bsm_data_target[:,:,0] = np.multiply(bsm_data_target[:,:,0], np.not_equal(bsm_data[:,:,0,0],0))
            bsm_data_target = bsm_data_target.reshape(bsm_data_target.shape[0],bsm_data_target.shape[1],bsm_data_target.shape[2],1)



            _, teacher_loss_sgn = teacher_loss(bsm_data_target, reco_sgn)
            # print("loss shape",teacher_loss_sgn.shape)
            # print(teacher_loss_bkg.shape)
            teacher_loss_sgn = teacher_loss_sgn.numpy()
            mse_teacher_tr = get_threshold(teacher_loss_bkg, 'mse ae')
            figure_of_merit = {
                    'teacher': get_metric(teacher_loss_bkg, teacher_loss_sgn),
                    'student': 0
                    }
            tprscore = []
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

            student_total_loss = []
            student_total_loss.append(predicted_loss)
            student_total_loss.append(predicted_loss_sgn)

            mse_student_tr = get_threshold(predicted_loss, 'mse ae')
            # figure_of_merit = {}
            figure_of_merit['student'] = get_metric(student_total_loss[0], student_total_loss[1])
            tprscore = []
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
            roc_student = axs[j].plot(figure_of_merit['student'][0], figure_of_merit['student'][1], "-",
                label=f'Cosearch student AUC = {figure_of_merit["student"][2]*100:.0f}%',
                linewidth=3, color=colors[K])
            K+=1


            roc_teacher = axs[j].plot(figure_of_merit['teacher'][0], figure_of_merit['teacher'][1], "-",
                label=f'teacher AUC = {figure_of_merit["teacher"][2]*100:.0f}%',
                linewidth=3, color='#016c59')

            axs[j].set_xlim(10**(-6),1)
            axs[j].set_ylim(10**(-6),1.2)
            axs[j].semilogx()
            axs[j].semilogy()
            axs[j].set_ylabel('True Positive Rate', )
            axs[j].set_xlabel('False Positive Rate', )
            axs[j].plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3)
            axs[j].vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
            axs[j].legend(loc='lower right', frameon=False, title=f'ROC {PLOTTING_LABELS[1:][j]}',fontsize='x-small' )
        fig.tight_layout()
        fig.savefig(result_dir + "/"+ m +"/ROC_sgn.pdf")

    return mse_student_tr, mse_teacher_tr,tprscore

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
    
    mse_teacher_tr = get_threshold(teacher_loss_bkg, 'mse ae')
    figure_of_merit = {
            'teacher': get_metric(teacher_loss_bkg, teacher_loss_sgn),
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
    plt.show()

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
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--distillation-loss', type=str, default='mse', help='Loss to use for distillation')
    # parser.add_argument('--best-student-list', type=str, help='Ordered list with students\' hyperparameters after tuning')
    parser.add_argument('--n-models', type=int, default=1, help='Actual # of models to build and analyze')
    parser.add_argument('--output-dir', type=str, default='output/tuning-analysis-plots/')

    args = parser.parse_args()
    #knowledge_distillation(**vars(args))
    #modlist = build_models(**vars(args))#args.list_models,args.input_train_file, args.input_test_file, args.input_val_file, args.input_signal_file,args.data_name, args.n_features, args.teacher_loss_name, args.batch_size, args.n_epochs, args.distillation_loss, args.n_models, args.output_dir)
    modlistres = build_models("output/l1_ae_test_loss.h5", "output/l1_ae_signal_loss.h5", "output/cotrain_results", "data", 3, 4096, None)

