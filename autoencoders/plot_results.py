import os
import h5py
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt


from plotting import (
    add_logo,
    reco_loss,
    read_loss_data,
    BSM_SAMPLES,
    PLOTTING_LABELS
    )

def get_metric(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    print(f"QCD+BSM {true_val.shape}")
    pred_val = np.concatenate((bsm, qcd))
    print(f"QCD+BSM predicted {pred_val.shape}")

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)

    return fpr_loss, tpr_loss, auc_data

def get_threshold(qcd, loss_type):

    qcd[::-1].sort()
    threshold = qcd[int(len(qcd)*10**-5)]
    print(loss_type, threshold)

    return threshold

def return_label(anomaly):
    if (anomaly == 'Leptoquark'):
        marker = 'o'; sample_label=r'LQ $\rightarrow$ b$\tau$'
    elif (anomaly == 'A to 4 leptons'):
        marker='X'; sample_label=r'A $\rightarrow$ 4$\ell$'
    elif anomaly == 'hToTauTau':
        marker = 'd'; sample_label=r'$h^{0} \rightarrow \tau\tau$'
    else:
        marker = 'v'; sample_label=r'$h^{\pm} \rightarrow \tau\nu$'
    return marker, sample_label

def plot_rocs(student_file, teacher_file, anomaly, color):

    # load AE model
    with h5py.File(student_file, 'r') as data:
        student_total_loss = []
        student_total_loss.append(data['predicted_loss'][:].flatten())
        for bsm in BSM_SAMPLES:
            student_total_loss.append(data['predicted_loss_'+ bsm][:].flatten())

    with h5py.File(teacher_file, 'r') as data:
        teacher_total_loss = []
        X_test_scaled = data['QCD'][:]
        qcd_prediction = data['predicted_QCD'][:]
        #compute loss
        teacher_total_loss.append(reco_loss(X_test_scaled, qcd_prediction.astype(np.float32), dense=False))
        #BSM
        for bsm in BSM_SAMPLES:
            bsm_target = data[bsm+'_scaled'][:]
            bsm_prediction = data['predicted_'+ bsm][:]
            teacher_total_loss.append(reco_loss(bsm_target, bsm_prediction.astype(np.float32), dense=False))

    marker, sample_label = return_label(anomaly)

    mse_student_tr = get_threshold(student_total_loss[0], 'mse ae')
    mse_teacher_tr = get_threshold(teacher_total_loss[0], 'mse ae')

    print('first student then teacher')

    if sample_label == r'LQ $\rightarrow$ b$\tau$':
        figure_of_merit = {
            'student': get_metric(student_total_loss[0], student_total_loss[1]),
            'teacher': get_metric(teacher_total_loss[0], teacher_total_loss[1])}
    elif sample_label == r'A $\rightarrow$ 4$\ell$':
        figure_of_merit = {
            'student': get_metric(student_total_loss[0], student_total_loss[2]),
            'teacher': get_metric(teacher_total_loss[0], teacher_total_loss[2])}
    elif sample_label == r'$h^{\pm} \rightarrow \tau\nu$':
        figure_of_merit = {
            'student': get_metric(student_total_loss[0], student_total_loss[3]),
            'teacher': get_metric(teacher_total_loss[0], teacher_total_loss[3])}
    elif sample_label == r'$h^{0} \rightarrow \tau\tau$':
        figure_of_merit = {
            'student': get_metric(student_total_loss[0], student_total_loss[4]),
            'teacher': get_metric(teacher_total_loss[0], teacher_total_loss[4])}

    print(f"figure_of_merit['student'][1] {figure_of_merit['student'][1].shape}")
    print(f"figure_of_merit['teacher'][1] {figure_of_merit['teacher'][1].shape}")

    plt.plot(np.arange(0,1), figure_of_merit['student'][1]/figure_of_merit['teacher'][1], "-",
        label=f'Student ROC / Teacher ROC (AUC ratio = {figure_of_merit["student"][2]/figure_of_merit["teacher"][2]:.3f}%)',
        linewidth=3, color=color)

    plt.xlim(10**(-6),1)
    plt.ylim(10**(-6),1.2)
    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate', )
    plt.xlabel('False Positive Rate', )
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=3)
    plt.vlines(1e-5, 0, 1, linestyles='--', color='#ef5675', linewidth=3)
    plt.legend(loc='lower right', frameon=False, title=f'ROC {sample_label}', )
    plt.tight_layout()

    return mse_student_tr, mse_teacher_tr

def plot_loss(ae_file, threshold, output_dir, label):
    # load AE model
    if label=='student':
        with h5py.File(ae_file, 'r') as data:
            baseline_total_loss = []
            baseline_total_loss.append(data['predicted_loss'][:].flatten())
            for bsm in BSM_SAMPLES:
                baseline_total_loss.append(data['predicted_loss_'+ bsm][:].flatten())
    else:
        baseline_total_loss, _, _, _ = read_loss_data(ae_file, 0.8)

    base = baseline_total_loss
    hrange = np.linspace(0,5000,100) if label=='teacher' else 100
    hrange = 100

    for i, base_bsm in enumerate(base):
        plt.hist(base_bsm, hrange,
            label=PLOTTING_LABELS[i],
            linewidth=3,
            color=colors[i],
            histtype='step',
            density=True)

    plt.semilogy()
    plt.ylabel('A.U.', )
    plt.xlabel('loss', )
    plt.vlines(threshold, 0, plt.gca().get_ylim()[1], linestyles='--', color='#ef5675', linewidth=3)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_{label}.pdf'))
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student', type=str)
    parser.add_argument('--teacher', type=str)
    parser.add_argument('--output-dir', type=str, default='plots/')
    args = parser.parse_args()

    colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']

    for bsm in BSM_SAMPLES:
        student_tr = plot_rocs(args.student, args.teacher, bsm, colors[0])
        plt.savefig(os.path.join(args.output_dir, f'student_rocs_{bsm}.pdf'))
        plt.clf()

    plot_loss(args.teacher, teacher_tr, args.output_dir, label='teacher')
    plot_loss(args.student, student_tr, args.output_dir, label='student')

