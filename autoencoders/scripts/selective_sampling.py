import os
import h5py
import numpy as np
import argparse
import shutil
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

matplotlib.rcParams.update({'font.size': 22,'figure.figsize':(7,8)})
fixed_seed = 2021
rng = np.random.default_rng(fixed_seed)


def selective_sampling_func(
        sample,
        bincenters,
        ratio_function):

    random_array = rng.random(len(sample))
    indecies = np.searchsorted(bincenters, sample,side='left')-1
    indecies = np.where(indecies==-1,0,indecies)
    indecies_to_keep = np.where(random_array<ratio_function[indecies])

    return indecies_to_keep


def fit_function(x, a, b):

    return a * np.exp(-b * x)


def main(args):

    background = np.load(args.data_file_bg)
    train_loss = background['teacher_train_loss']
    train_data = background['x_train']

    signal = np.load(args.data_file_sig)
    signal_loss = signal[f'teacher_loss_{args.signal_name}']
    signal_data = signal[f'bsm_data_{args.signal_name}']

    values, bins = np.histogram(train_loss, bins=100)
    bincenters = 0.5*(bins[1:]+bins[:-1])

    threshold = args.fit_threshold
    fit_range_mask = bincenters>threshold
    popt, pcov = curve_fit(fit_function,
        xdata=bincenters[fit_range_mask],
        ydata=values[fit_range_mask])

    ratio_function = fit_function(bincenters, *popt)/values
    ratio_function[np.where(ratio_function>1.)[0][0]:]=1.

    indecies_to_keep = selective_sampling_func(train_loss,bincenters,ratio_function)

    if args.signal_fraction > 0:
        indecies_to_keep_signal = rng.choice(len(signal_loss), int(args.signal_fraction*len(signal_loss)),
            replace=False)


    fig, axs = plt.subplots(2,figsize=(12,15), gridspec_kw={'height_ratios': [3, 1]})
    _ = axs[0].hist(train_loss,bins=100,histtype='step',linewidth=3,color='#1845fb',label='data')
    axs[0].hist(train_loss[indecies_to_keep],bins=100,color='#3f90da',label='subsampled data')
    axs[0].plot(bincenters, fit_function(bincenters, *popt), '#ff5e02',linewidth=3,
         label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    axs[0].semilogy()
    axs[0].set_ylabel('A.U.')
    axs[0].legend()
    _ = axs[1].plot(bincenters,ratio_function,'#ff5e02',linewidth=3)
    axs[1].axhline(1., color='black',linestyle='--',linewidth=3)
    axs[1].semilogy()
    axs[1].set_ylabel('Ratio')
    axs[1].set_xlabel('Teacher Loss')
    plt.savefig(os.path.join(args.plot_dir, f'train_loss_fit.pdf'), bbox_inches="tight")
    plt.clf()


    mask_bg_discarded = np.ones(len(train_loss), dtype=bool)
    mask_bg_discarded[indecies_to_keep] = False

    np.savez(args.outfile_train_loss_bg,
        teacher_loss=train_loss[indecies_to_keep],
        data=train_data[indecies_to_keep])

    np.savez(args.outfile_discarded_test_loss,
        teacher_loss=train_loss[mask_bg_discarded],
        data=train_data[mask_bg_discarded])

    # If we are also combining with signal, save the combination and the remaining signal for validation
    if args.signal_fraction > 0:
        teacher_loss_bg_plus_sig = np.concatenate((train_loss[indecies_to_keep],signal_loss[indecies_to_keep_signal]), axis=0)
        teacher_data_bg_plus_sig = np.concatenate((train_data[indecies_to_keep],signal_data[indecies_to_keep_signal]), axis=0)
        np.savez(args.outfile_train_loss_bg_sig,
            teacher_loss=teacher_loss_bg_plus_sig,
            data=teacher_data_bg_plus_sig)

        mask_signal_for_valid = np.ones(len(signal_loss), dtype=bool)
        mask_signal_for_valid[indecies_to_keep_signal] = False

        shutil.copyfile(args.data_file_sig, args.outfile_signal_loss) #suboptimal, but I do not want to overwrite the file. to be fixed.
        old_data = np.load(args.data_file_sig)
        np.savez(args.outfile_signal_loss,
            **old_data,
            f'teacher_loss_{args.signal_name}'=signal_loss[mask_signal_for_valid],
            f'bsm_data_{args.signal_name}'=signal_data[mask_signal_for_valid])

        fig, axs = plt.subplots(2,figsize=(12,15), gridspec_kw={'height_ratios': [3, 1]})
        values_combi, bins, patches = axs[0].hist(
            teacher_loss_bg_plus_sig,
            bins=100,
            histtype='step',
            color='r',
            linewidth=3,
            label='BG + {} {}'.format(args.signal_fraction,args.signal_name))

        values_bg, bins, patches = axs[0].hist(
            train_loss[indecies_to_keep],
            bins=100,
            histtype='step',
            color='b',
            linewidth=3,
            label='BG')

        bincenters = 0.5*(bins[1:]+bins[:-1])
        axs[0].semilogy()
        axs[0].legend()
        axs[0].set_title('Training sample')
        axs[0].set_ylabel('A.U.')

        _ = axs[1].plot(bincenters,values_combi/values_bg,'r',linewidth=3)
        axs[1].axhline(1., color='black',linestyle='--',linewidth=3)
        axs[1].set_ylabel('Ratio')
        axs[1].set_xlabel('Teacher Loss')
        plt.savefig(os.path.join(args.plot_dir, 'final_training_sample_with_signal_{}{}.pdf'.format(str(args.signal_fraction).replace('.','_'),\
                                                                                            args.signal_name.replace(' ','_'))), bbox_inches="tight")
        plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('data_file_bg', type=str,
        help='Where is the data for bg')
    parser.add_argument('data_file_sig', type=str,
        help='Where is the data for sig')

    parser.add_argument('--signal-name', type=str,
        help='signal name')
    parser.add_argument('--signal-fraction', type=float, default=0.25,
        help='signal fraction')
    parser.add_argument('--fit-threshold', type=float,default=1.2,
        help='threshold above which to run the fit')
    parser.add_argument('--outfile-train-loss-bg', type=str,
        help='Output file')
    parser.add_argument('--outfile-train-loss-bg-sig', type=str,
        help='Output file')
    parser.add_argument('--outfile-discarded-test-loss', type=str,
        help='Output file')
    parser.add_argument('--outfile-signal-loss', type=str,
        help='Output file')
    parser.add_argument('--plot-dir', type=str,
        help='plotting directory')

    args = parser.parse_args()
    main(args)