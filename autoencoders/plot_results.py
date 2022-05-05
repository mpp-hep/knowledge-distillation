import os
import scipy
import h5py
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']
SAMPLES = ['QCD']+BSM_SAMPLES

PLOTTING_LABELS = ['Background', r'LQ $\rightarrow$ b$\tau$', r'A $\rightarrow$ 4$\ell$',
    r'$h^{\pm} \rightarrow \tau\nu$', r'$h^{0} \rightarrow \tau\tau$']
LABELS = {
    'Leptoquark': (r'LQ $\rightarrow$ b$\tau$', 'o', '#016c59'),
    'A to 4 leptons': (r'A $\rightarrow$ 4$\ell$', 'X', '#7a5195'),
    'hChToTauNu': (r'$h^{\pm} \rightarrow \tau\nu$', 'v', '#67a9cf'),
    'hToTauTau': (r'$h^{0} \rightarrow \tau\tau$', 'd', '#ffa600')}

def get_metric(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_data = auc(fpr_loss, tpr_loss)

    return fpr_loss, tpr_loss, auc_data


def get_threshold(qcd, loss_type):

    qcd[::-1].sort()
    threshold = qcd[int(len(qcd)*10**-5)]

    return threshold


def make_plot_training_history(input_file, output_dir):

    with h5py.File(input_file, 'r') as h5f:
        loss = h5f['loss'][:]
        val_loss = h5f['val_loss'][:]

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Training History')

    plt.semilogy()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_hist.pdf'))


def plot_rocs(student_file, teacher_file, signal_file, teacher_loss_name, anomaly):

    # load AE model
    with h5py.File(student_file, 'r') as data:
        student_total_loss = []
        student_total_loss.append(data['predicted_loss'][:].flatten())
        student_total_loss.append(data[f'predicted_loss_{anomaly[0]}'][:].flatten())

    with h5py.File(teacher_file, 'r') as data:
        teacher_total_loss = []
        teacher_total_loss.append(np.array(data[teacher_loss_name])[:].flatten())

    with h5py.File(signal_file, 'r') as bsm_data:
        # for graph anomalies loss are all in one array
        if 'ProcessID' in bsm_data.keys():
            teacher_total_loss.append(bsm_data[teacher_loss_name][bsm_data['ProcessID'][:,0]==anomaly[1]].flatten())
        else:
            teacher_total_loss.append(bsm_data[f'{teacher_loss_name}_{anomaly[0]}'][:].flatten())

    mse_student_tr = get_threshold(student_total_loss[0], 'mse ae')
    mse_teacher_tr = get_threshold(teacher_total_loss[0], 'mse ae')

    figure_of_merit = {
        'student': get_metric(student_total_loss[0], student_total_loss[1]),
        'teacher': get_metric(teacher_total_loss[0], teacher_total_loss[1])}

    plt.plot(figure_of_merit['student'][0], figure_of_merit['student'][1], "-",
        label=f'student AUC = {figure_of_merit["student"][2]*100:.0f}%',
        linewidth=3, color=colors[0])

    plt.plot(figure_of_merit['teacher'][0], figure_of_merit['teacher'][1], "-",
        label=f'teacher AUC = {figure_of_merit["teacher"][2]*100:.0f}%',
        linewidth=3, color=colors[1])

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

    return mse_student_tr, mse_teacher_tr

def plot_loss_pull(student_file, teacher_file, signal_file, teacher_loss_name, output_dir):
    # load AE model
    with h5py.File(student_file, 'r') as data:
        student_total_loss = []
        student_total_loss.append(data['predicted_loss'][:].flatten())
        for bsm in BSM_SAMPLES:
            student_total_loss.append(data[f'predicted_loss_{bsm}'][:].flatten())

    with h5py.File(teacher_file, 'r') as data:
        teacher_total_loss = []
        teacher_total_loss.append(np.array(data[teacher_loss_name])[:].flatten())

    with h5py.File(signal_file, 'r') as bsm_data:
        # for graph anomalies loss are all in one array
        for bsm in zip(BSM_SAMPLES, [33,30,31,32]):
            if 'ProcessID' in bsm_data.keys():
                teacher_total_loss.append(bsm_data[teacher_loss_name][bsm_data['ProcessID'][:,0]==bsm[1]].flatten())
            else:
                teacher_total_loss.append(bsm_data[f'{teacher_loss_name}_{bsm[0]}'][:].flatten())

    hrange = np.linspace(0,5000,100)
    hrange = 100

    for i, base in enumerate(zip(teacher_total_loss,student_total_loss)):
        diff = np.subtract(base[0], base[1])
        skew = scipy.stats.skew(diff)
        kurt = scipy.stats.kurtosis(diff)
        plt.hist(diff,
            hrange,
            label=PLOTTING_LABELS[i]+f' skew={skew:.0f}; kurt={kurt:.0f}',
            linewidth=3,
            color=colors[i],
            histtype='step',
            density=True)

    plt.semilogy()
    plt.ylabel('A.U.', )
    plt.xlabel('teacher loss - student loss', )
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'loss_pull.pdf'))
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student', type=str)
    parser.add_argument('--teacher', type=str)
    parser.add_argument('--teacher-loss-name', type=str)
    parser.add_argument('--signal', type=str)
    parser.add_argument('--output-dir', type=str, default='plots/')
    args = parser.parse_args()

    colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']

    make_plot_training_history(args.student, args.output_dir)

    for bsm in zip(BSM_SAMPLES, [33,30,31,32]):
        student_tr, teacher_tr = plot_rocs(args.student, args.teacher, args.signal,
            args.teacher_loss_name, bsm)
        plt.savefig(os.path.join(args.output_dir, f'student_rocs_{bsm[0]}.pdf'))
        plt.clf()

    plot_loss_pull(args.student, args.teacher, args.signal, args.teacher_loss_name, args.output_dir)

