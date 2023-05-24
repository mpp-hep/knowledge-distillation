import os
import scipy
import h5py
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']
SAMPLES = ['Background']+BSM_SAMPLES
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


def plot_rocs(
        student_file,
        teacher_file,
        signal_file,
        teacher_loss_name,
        anomaly):

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

    roc_student = plt.plot(figure_of_merit['student'][0], figure_of_merit['student'][1], "-",
        label=f'student AUC = {figure_of_merit["student"][2]*100:.0f}%',
        linewidth=3, color=colors[0])

    roc_teacher = plt.plot(figure_of_merit['teacher'][0], figure_of_merit['teacher'][1], "-",
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

def plot_loss(
        student_file,
        teacher_file,
        signal_file,
        teacher_loss_name,
        output_dir):
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

    hrange = np.linspace(0,50,500)

    for i, base in enumerate(zip(teacher_total_loss,student_total_loss)):
        plt.hist(base[0],
            hrange,
            label=PLOTTING_LABELS[i],
            linewidth=3,
            color=colors[i],
            histtype='step',
            density=True)

    plt.semilogy()
    plt.ylabel('A.U.', )
    plt.xlabel('teacher loss', )
    plt.legend(loc=(1.04,0))
    plt.savefig(os.path.join(output_dir, f'loss_teacher.pdf'), bbox_inches="tight")
    plt.clf()


    hrange = np.linspace(0,50,500)

    for i, base in enumerate(zip(teacher_total_loss,student_total_loss)):
        plt.hist(base[1],
            hrange,
            label=PLOTTING_LABELS[i],
            linewidth=3,
            color=colors[i],
            histtype='step',
            density=True)

    plt.semilogy()
    plt.ylabel('A.U.', )
    plt.xlabel('student loss', )
    plt.legend(loc=(1.04,0))
    plt.savefig(os.path.join(output_dir, f'loss_student.pdf'), bbox_inches="tight")
    plt.clf()

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
        # Mean, Median, STD, 16%quantile, 84%quantile, Skew, Kurtotis
        diff = np.subtract(base[0], base[1])
        mean = np.mean(diff)
        median = np.median(diff)
        std = np.std(diff)
        quant16 = np.quantile(diff, 0.16)
        quant84 = np.quantile(diff, 0.84)
        skew = scipy.stats.skew(diff)
        kurt = scipy.stats.kurtosis(diff)
        plt.hist(diff,
            hrange,
            label=PLOTTING_LABELS[i]+\
            f'\n mu={mean:.1f} median={median:.2f} std={std:.1f}\n quant16={quant16:.3f} quant84={quant84:.3f}\n skew={skew:.0f} kurt={kurt:.0f}',
            linewidth=3,
            color=colors[i],
            histtype='step',
            density=True)

    plt.semilogy()
    plt.ylabel('A.U.', )
    plt.xlabel('teacher loss - student loss', )
    plt.legend(loc=(1.04,0))
    plt.savefig(os.path.join(output_dir, f'loss_pull.pdf'), bbox_inches="tight")
    plt.clf()


def compute_profile(x, y, nbin):
    """
    Returns the center of bins array, the mean of y for each bin and stand.dev.
    https://vmascagn.web.cern.ch/LABO_2020/profile_plot.html
    """
    # use of the 2d hist by numpy to avoid plotting
    h, xe, ye = np.histogram2d(x,y,nbin)

    # bin width
    xbinw = xe[1]-xe[0]

    # getting the mean and RMS values of each vertical slice of the 2D distribution
    # also the x valuse should be recomputed because of the possibility of empty slices
    x_array      = []
    x_slice_mean = []
    x_slice_rms  = []
    for i in range(xe.size-1):
        yvals = y[ (x>xe[i]) & (x<=xe[i+1]) ]
        if yvals.size>0: # do not fill the quanties for empty slices
            x_array.append(xe[i]+ xbinw/2)
            x_slice_mean.append( yvals.mean())
            x_slice_rms.append( yvals.std())
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)

    return x_array, x_slice_mean, x_slice_rms


def make_profile_plot(
        student_file,
        teacher_file,
        signal_file,
        teacher_loss_name,
        output_dir,
        nbins=(100,100)):

    # load AE model
    with h5py.File(student_file, 'r') as data:
        student_loss = []
        student_loss.append(np.array(data['predicted_loss'][:].flatten()))
        for bsm in BSM_SAMPLES:
            student_loss.append(np.array(data[f'predicted_loss_{bsm}'][:]).flatten())

    with h5py.File(teacher_file, 'r') as data:
        teacher_loss = []
        teacher_loss.append(np.array(data['teacher_loss'])[:].flatten())

    with h5py.File(signal_file, 'r') as bsm_data:
        # for graph anomalies loss are all in one array
        for bsm in zip(BSM_SAMPLES, [33,30,31,32]):
            if 'ProcessID' in bsm_data.keys():
                teacher_loss.append(bsm_data[teacher_loss_name][bsm_data['ProcessID'][:,0]==bsm[1]].flatten())
            else:
                teacher_loss.append(np.array(bsm_data[f'teacher_loss_{bsm[0]}'][:]).flatten())

    for i, samp in enumerate(SAMPLES):
        # compute the profile
        p_x, p_mean, p_rms = compute_profile(teacher_loss[i],student_loss[i],nbins)
        plt.errorbar(p_x, p_mean, p_rms, fmt='_', ecolor='orange', color='orange')
        plt.hist2d(teacher_loss[i], student_loss[i], 100, cmap='GnBu',
            norm=matplotlib.colors.LogNorm())
        plt.title(samp)
        plt.ylabel('Student loss')
        plt.xlabel('Teacher loss')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f'profile_{samp.lower()}.pdf'))
        plt.clf()


def prepare_violin_input(x, y, nbin=10):
    """
    Returns the center of bins array, the mean of y for each bin and stand.dev.
    https://vmascagn.web.cern.ch/LABO_2020/profile_plot.html
    """
    # use of the 2d hist by numpy to avoid plotting
    h, xe, ye = np.histogram2d(x,y,nbin)

    # bin width
    xbinw = xe[1]-xe[0]

    # getting the mean and RMS values of each vertical slice of the 2D distribution
    # also the x valuse should be recomputed because of the possibility of empty slices
    x_array      = []
    x_violin = []
    for i in range(xe.size-1):
        yvals = y[ (x>xe[i]) & (x<=xe[i+1]) ]
        x_array.append(xe[i]+ xbinw/2)
        x_violin.append(yvals)
    x_array = np.array(x_array)
    x_violin = np.array(x_violin)

    return x_array, x_violin


def make_violin_plot(
        student_file,
        teacher_file,
        signal_file,
        output_dir,
        nbins=10):

    # load AE model
    with h5py.File(student_file, 'r') as data:
        student_loss = []
        student_loss.append(np.array(data['predicted_loss'][:].flatten()))
        for bsm in BSM_SAMPLES:
            student_loss.append(np.array(data[f'predicted_loss_{bsm}'][:]).flatten())

    with h5py.File(teacher_file, 'r') as data:
        teacher_loss = []
        teacher_loss.append(np.array(data['teacher_train_loss'])[:].flatten())

    with h5py.File(signal_file, 'r') as bsm_data:
        # for graph anomalies loss are all in one array
        for bsm in zip(BSM_SAMPLES, [33,30,31,32]):
            if 'ProcessID' in bsm_data.keys():
                teacher_loss.append(bsm_data['teacher_loss'][bsm_data['ProcessID'][:,0]==bsm[1]].flatten())
            else:
                teacher_loss.append(np.array(bsm_data[f'{'teacher_loss'}_{bsm[0]}'][:]).flatten())

    for i, samp in enumerate(SAMPLES):
        p_x, p_violin = prepare_violin_input(teacher_loss[i],student_loss[i], nbins)
        # compute the profile
        plt.violinplot(p_violin, p_x, widths=p_x[0])
        plt.title(samp)
        plt.ylabel('Student loss')
        plt.xlabel('Teacher loss')
        plt.savefig(os.path.join(output_dir, f'violin_{samp.lower()}.pdf'))
        plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('student', type=str)
    parser.add_argument('teacher', type=str)

    parser.add_argument('--teacher-loss-name', type=str)
    parser.add_argument('--signal', type=str)
    parser.add_argument('--output-dir', type=str, default='plots/')

    args = parser.parse_args()

    colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']

    make_profile_plot(
        args.student,
        args.teacher,
        args.signal,
        args.output_dir,
        nbins=(50,50))

    make_violin_plot(
        args.student,
        args.teacher,
        args.signal,
        args.output_dir,
        nbins=4)

    make_plot_training_history(args.student, args.output_dir)
    plt.clf()

    for bsm in zip(BSM_SAMPLES, [33,30,31,32]):
        student_tr, teacher_tr = plot_rocs(args.student, args.teacher, args.signal, bsm)
        plt.savefig(os.path.join(args.output_dir, f'student_rocs_{bsm[0]}.pdf'))
        plt.clf()

    plot_loss_pull(args.student, args.teacher, args.signal, args.output_dir)

    plot_loss(args.student, args.teacher, args.signal, args.output_dir)
