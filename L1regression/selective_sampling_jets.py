import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import shutil 
from utils.data_processing import idx_particle_type,idx_feature
import utils.data_processing  as data_proc
matplotlib.use("Agg") 
import mplhep as hep
hep.style.use(hep.style.CMS) 

fixed_seed = 2021
rng = np.random.default_rng(fixed_seed)

def selective_sampling_func(sample,bincenters,ratio_function):
    random_array = rng.random(len(sample))
    indecies = np.searchsorted(bincenters, sample,side='left')-1 
    indecies = np.where(indecies==-1,0,indecies)
    indecies_to_keep = np.where(random_array<ratio_function[indecies])
    return indecies_to_keep


def check_dataset_proportions(txt_outfile,jets_data,idx_jet):
    values, counts = np.unique(np.sum(jets_data[:,idx_jet[0]:idx_jet[1],0]!=0,axis=1), return_counts=True)
    tot_evt = np.sum(counts)
    with open(txt_outfile, "w") as f:
        print('Number of events with jets == N', file=f)
        for v,c in zip(values,counts):
            print("{} jets : ".format(v),"{} events, ".format(c),"fraction %.4f "%(c/tot_evt), file=f)
        print('Number of events with jets >= N', file=f)
        for i,(v,c) in enumerate(zip(values,counts)):
            print(">={} jets : ".format(v),"{} events, ".format(np.sum(counts[i:])),"fraction %.4f "%(np.sum(counts[i:])/tot_evt), file=f)



def downsample(data,njets,fraction_to_keep,idx_jet,jet_pt_idx,mode="=="):
    indecies_of_njets = np.where(eval('np.sum(data[:,idx_jet[0]:idx_jet[1]+1,jet_pt_idx]!=0,axis=1){}njets'.format(mode)))[0]
    indecies_to_train = rng.choice(indecies_of_njets, int(fraction_to_keep*len(indecies_of_njets)),replace=False)
    indecies_to_test = np.ones(len(data), dtype=bool)
    indecies_to_test[indecies_to_train] = False
    return indecies_to_train,indecies_to_test
    
    
def downsample_training(data,idx_jet,jet_pt_idx,fractions_to_keep,mode="=="):
    num_jets_selections = len(fractions_to_keep)
    indecies_to_train_list = []
    for jet_num, fraction in enumerate(fractions_to_keep):
        indecies_to_train_step,_ = downsample(data,njets=jet_num,mode=mode,fraction_to_keep=fractions_to_keep[jet_num],idx_jet=idx_jet,jet_pt_idx=jet_pt_idx)
        indecies_to_train_list.append(indecies_to_train_step)
    indecies_to_train_otherjets = np.where(np.sum(data[:,idx_jet[0]:idx_jet[1]+1,jet_pt_idx]!=0,axis=1)>=num_jets_selections)[0]
    indecies_to_train_list.append(indecies_to_train_otherjets)
    indecies_to_train = np.concatenate(indecies_to_train_list,axis=0)
    indecies_to_test = np.ones(len(data), dtype=bool)
    indecies_to_test[indecies_to_train] = False
    return indecies_to_train, indecies_to_test 

def main_selective_sampling_jets(data_file='',proc_names='',fractions_to_keep=[1.], mode="==", outfile='',sampling_var='',sampling_threshold=0, outfile_discarded='',txt_outfile='',plot_dir=''):
    """
    Performs selective sub sampling, and saves files with selected events and discarded events
    Arguments:
        data_file: str, path to the input file 
        proc_names: list of str, processes to keep
        fractions_to_keep: list, fraction of jets to keep, starting from 0 jets. E.g. [1.,0.2] : will keep 100% of events with 0 jets, and only 20% of events with 1 jet
        sampling_var: str, which variable apply sampling on (true/original MET or HT, or ratio), if empty - no sampling is applied
        sampling_threshold: float, threshold form when to flatten MET/HT spectrum (sampling_var)
        outfile : str, path to the output file, for events used for the training
        outfile_discarded : str, path to the output file, for events not used for the training
        txt_outfile : str, path to the output textfile where to write the summary printout
        plot_dir: str, path to plotting directory
    """

    with h5py.File(data_file,'r') as open_file :
        ids = np.array(open_file['ids'])
        ids_names = np.array(open_file['ids_names'])
        ids_names_dict = data_proc.get_process_id_dict(ids_names)

        proc_selection_str = ''
        for proc in proc_names:
            proc_selection_str+='(ids==%f)|'%ids_names_dict[proc]
        proc_selection_str=proc_selection_str[:-1] # remove the last "|"
        mask = np.where(eval(proc_selection_str))
        
        reco_data = np.array(open_file['smeared_data'])[mask]
        reco_met = np.array(open_file['smeared_met'])[mask]
        reco_ht = np.array(open_file['smeared_ht'])[mask]
        true_data = np.array(open_file['true_data'])[mask]
        true_met = np.array(open_file['true_met'])[mask]
        true_ht = np.array(open_file['true_ht'])[mask]
        original_met = np.array(open_file['original_met'])[mask]
        ids = np.array(open_file['ids'])[mask]

    idx_jet = idx_particle_type['jet']
    pt_idx = int(idx_feature['pt'])
    indecies_to_train,indecies_to_test = downsample_training(data=reco_data,idx_jet=idx_jet,jet_pt_idx=pt_idx, fractions_to_keep=fractions_to_keep,mode=mode)
    check_dataset_proportions(txt_outfile,reco_data,idx_jet)
    check_dataset_proportions(txt_outfile.replace('.txt','_after_subsampling.txt'),reco_data[indecies_to_train],idx_jet)

    sampling_var_dict = {}
    sampling_var_dict['true_met'] = true_met
    sampling_var_dict['original_met'] = original_met
    sampling_var_dict['true_ht'] = true_ht
    sampling_var_dict['true_ht_over_reco_ht'] = np.where(reco_ht!=0,true_ht/reco_ht,0.)
    sampling_var_dict['true_met_over_reco_met'] = np.where(reco_met!=0,true_met/reco_met,0.)
    sampling_var_dict['original_met_over_reco_met'] = np.where(reco_met!=0,original_met/reco_met,0.)
    if sampling_var in list(sampling_var_dict.keys()):
        max_var = np.max(sampling_var_dict[sampling_var][indecies_to_train])
        values, bins = np.histogram(sampling_var_dict[sampling_var][indecies_to_train],bins=np.linspace(0,max_var,200))
        bincenters = 0.5*(bins[1:]+bins[:-1])
        ratio_function = values[np.searchsorted(bincenters,sampling_threshold)]/values
        ratio_function[:np.argmin(ratio_function)] = min(ratio_function)
        indecies_to_keep_met = selective_sampling_func(sampling_var_dict[sampling_var][indecies_to_train],bincenters,ratio_function)
        final_indecies_to_keep_met = indecies_to_train[indecies_to_keep_met]
        mask_met_discarded = np.ones(len(sampling_var_dict[sampling_var]), dtype=bool)
        mask_met_discarded[final_indecies_to_keep_met] = False

        fig, axs = plt.subplots(2,figsize=(12,15), gridspec_kw={'height_ratios': [3, 1]})
        _ = axs[0].hist(sampling_var_dict[sampling_var][indecies_to_train],bins=100,histtype='step',linewidth=3,color='#1845fb',label='data',density=False)
        axs[0].hist(sampling_var_dict[sampling_var][indecies_to_train][indecies_to_keep_met],bins=100,color='#3f90da',label='subsampled data',density=False)
        axs[0].semilogy()
        axs[0].set_ylabel('A.U.')
        axs[0].legend()
        _ = axs[1].plot(bincenters,ratio_function,'#ff5e02',linewidth=3)
        axs[1].axhline(1., color='black',linestyle='--',linewidth=3)
        axs[1].semilogy()
        axs[1].set_ylabel('Ratio')
        axs[1].set_xlabel('Yield')
        plt.savefig(os.path.join(plot_dir, f'sampling_{sampling_var}.pdf'), bbox_inches="tight")
        plt.clf()
        indecies_to_train = final_indecies_to_keep_met
        indecies_to_test = mask_met_discarded

    for file,indecies in zip([outfile,outfile_discarded],[indecies_to_train,indecies_to_test]):
        with h5py.File(file, 'w') as h5f:
            h5f.create_dataset('smeared_data', data=reco_data[indecies])
            h5f.create_dataset('smeared_met', data=reco_met[indecies])  
            h5f.create_dataset('smeared_ht', data=reco_ht[indecies])    
            h5f.create_dataset('true_data', data=true_data[indecies])    
            h5f.create_dataset('true_met', data=true_met[indecies])    
            h5f.create_dataset('true_ht', data=true_ht[indecies])    
            h5f.create_dataset('original_met', data=original_met[indecies])             
            h5f.create_dataset('ids', data=ids[indecies])    
            h5f.create_dataset('ids_names', data=ids_names)     

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Where is the data')
    parser.add_argument('--fractions_to_keep', type=str, default='0.', help='Commma-separated fractions of jets to keep, starting from 0 jets and increasing by 1 jets : e.g. 0,0.5 will remove all events with 0 jets, and keep 50 percent of events with 1 jet')
    parser.add_argument('--mode', type=str, default='==',help='Mode for jets selection : == or >=')
    parser.add_argument('--sampling_var', type=str, default='', help='Variable to apply sampling on (true/original MET or HT)')
    parser.add_argument('--sampling_threshold', type=float, default=180, help='MET or HT value threshold for sampling')
    parser.add_argument('--outfile', type=str, help='Output file with events for the training')
    parser.add_argument('--outfile_discarded', type=str, help='Output file with discarded events not used for the training')
    parser.add_argument('--proc_names', type=str, default='QCD,W,hChToTauNu,hToTauTau',help='Processes to keep')
    parser.add_argument('--txt_outfile', type=str, help='Output txt file for printout ')
    parser.add_argument('--plot_dir', type=str, help='Plotting dir')
    args = parser.parse_args()
    args.fractions_to_keep = [float(f) for f in args.fractions_to_keep.replace(' ','').split(',')]
    args.proc_names = [f for f in args.proc_names.replace(' ','').split(',')]
    main_selective_sampling_jets(**vars(args))




