import h5py
import numpy as np
import matplotlib
import argparse
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import shutil
matplotlib.use("Agg") 

matplotlib.rcParams.update({'font.size': 22,'figure.figsize':(7,8)})
fixed_seed = 2021
rng = np.random.default_rng(fixed_seed)

def selective_sampling_func(sample,bincenters,ratio_function):
    random_array = rng.random(len(sample))
    indecies = np.searchsorted(bincenters, sample,side='left')-1 
    indecies = np.where(indecies==-1,0,indecies)
    indecies_to_keep = np.where(random_array<ratio_function[indecies])
    return indecies_to_keep


def fit_function(x, a, b):
    return a * np.exp(-b * x)


def main_selective_sampling(data_file_bg='',data_file_sig='',signal_name='',signal_fraction=0.,fit_threshold=0.,plot_dir='',
                            outfile_train_loss_bg='',outfile_train_loss_bg_sig='',outfile_discarded_test_loss='',outfile_signal_loss=''):

    with h5py.File(data_file_bg,'r') as open_file :
        train_loss = np.array(open_file['teacher_loss'])
        train_data = np.array(open_file['data'])
    with h5py.File(data_file_sig,'r') as open_file :
        signal_loss = np.array(open_file['teacher_loss_%s'%signal_name])
        signal_data = np.array(open_file['bsm_data_%s'%signal_name])


    values, bins = np.histogram(train_loss,bins=100)
    bincenters = 0.5*(bins[1:]+bins[:-1])

    threshold = fit_threshold
    fit_range_mask = bincenters>threshold
    popt, pcov = curve_fit(fit_function, xdata=bincenters[fit_range_mask], ydata=values[fit_range_mask])

    ratio_function = fit_function(bincenters, *popt)/values
    ratio_function[np.where(ratio_function>1.)[0][0]:]=1.

    indecies_to_keep = selective_sampling_func(train_loss,bincenters,ratio_function)

    if signal_fraction > 0:
        indecies_to_keep_signal = rng.choice(len(signal_loss), int(signal_fraction*len(signal_loss)),replace=False)


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
    plt.savefig(os.path.join(plot_dir, f'train_loss_fit.pdf'), bbox_inches="tight")
    plt.clf()


    mask_bg_discarded = np.ones(len(train_loss), dtype=bool)
    mask_bg_discarded[indecies_to_keep] = False

    with h5py.File(outfile_train_loss_bg, 'w') as h5f:
        h5f.create_dataset('teacher_loss', data=train_loss[indecies_to_keep])
        h5f.create_dataset('data', data=train_data[indecies_to_keep])    
    
    with h5py.File(outfile_discarded_test_loss,'w') as h5f:
        h5f.create_dataset('teacher_loss',data=train_loss[mask_bg_discarded])    
        h5f.create_dataset('data',data=train_data[mask_bg_discarded])        

    #If we are also combining with signal, save the combination and the remaining signal for validation
    if signal_fraction > 0:    
        teacher_loss_bg_plus_sig = np.concatenate((train_loss[indecies_to_keep],signal_loss[indecies_to_keep_signal]), axis=0)
        teacher_data_bg_plus_sig = np.concatenate((train_data[indecies_to_keep],signal_data[indecies_to_keep_signal]), axis=0)
        with h5py.File(outfile_train_loss_bg_sig, 'w') as h5f:
            h5f.create_dataset('teacher_loss', data=teacher_loss_bg_plus_sig)
            h5f.create_dataset('data', data=teacher_data_bg_plus_sig)
        
        mask_signal_for_valid = np.ones(len(signal_loss), dtype=bool)
        mask_signal_for_valid[indecies_to_keep_signal] = False 

        shutil.copyfile(data_file_sig, outfile_signal_loss) #suboptimal, but I do not want to overwrite the file. to be fixed.
        with h5py.File(outfile_signal_loss,'r+') as h5f:
            del h5f['teacher_loss_%s'%signal_name] 
            del h5f['bsm_data_%s'%signal_name] 
            h5f.create_dataset('teacher_loss_%s'%signal_name,data=signal_loss[mask_signal_for_valid])    
            h5f.create_dataset('bsm_data_%s'%signal_name,data=signal_data[mask_signal_for_valid])    
    
        fig, axs = plt.subplots(2,figsize=(12,15), gridspec_kw={'height_ratios': [3, 1]})
        values_combi, bins, patches = axs[0].hist(teacher_loss_bg_plus_sig,bins=100,histtype='step',color='r',linewidth=3,label='BG + {} {}'.format(signal_fraction,signal_name))
        values_bg, bins, patches = axs[0].hist(train_loss[indecies_to_keep],bins=100,histtype='step',color='b',linewidth=3,label='BG')
        bincenters = 0.5*(bins[1:]+bins[:-1])
        axs[0].semilogy()
        axs[0].legend()
        axs[0].set_title('Training sample')
        axs[0].set_ylabel('A.U.')

        _ = axs[1].plot(bincenters,values_combi/values_bg,'r',linewidth=3)
        axs[1].axhline(1., color='black',linestyle='--',linewidth=3)
        axs[1].set_ylabel('Ratio')
        axs[1].set_xlabel('Teacher Loss')
        plt.savefig(os.path.join(plot_dir, 'final_training_sample_with_signal_{}{}.pdf'.format(str(signal_fraction).replace('.','_'),\
                                                                                            signal_name.replace(' ','_'))), bbox_inches="tight")
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file-bg', type=str, help='Where is the data for bg')
    parser.add_argument('--data-file-sig', type=str, help='Where is the data for sig')
    parser.add_argument('--signal-name', type=str, help='signal name')
    parser.add_argument('--signal-fraction', type=float, default=0.25, help='signal fraction')
    parser.add_argument('--fit-threshold', type=float,default=1.2, help='threshold above which to run the fit')
    parser.add_argument('--outfile-train-loss-bg', type=str, help='Output file')
    parser.add_argument('--outfile-train-loss-bg-sig', type=str, help='Output file')
    parser.add_argument('--outfile-discarded-test-loss', type=str, help='Output file')
    parser.add_argument('--outfile-signal-loss', type=str, help='Output file')
    parser.add_argument('--plot-dir', type=str, help='plotting directory')
    args = parser.parse_args()
    main_selective_sampling(**vars(args))




