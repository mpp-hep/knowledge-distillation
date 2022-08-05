import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import shutil 
import pickle
import pandas as pd
import vector
import copy
from scipy.optimize import curve_fit
from utils.data_processing import idx_particle_type,idx_feature
matplotlib.use("Agg") 
import mplhep as hep
hep.style.use(hep.style.CMS) 

fixed_seed = 2021
rng = np.random.default_rng(fixed_seed)
markers=['s', 'o', 'D', 'v']
colors=['g','black','orange','b']
        
def energy_res_function(x, a, b,c):
    return np.where(x>0,np.sqrt(a/np.power(x,2)+b/x+c),0.)
    
def get_jer(pt,eta,params,eta_bin=3.2):
    return np.where(eta<eta_bin,energy_res_function(pt, *params[0]),energy_res_function(pt, *params[1]))


def get_em_correction(eta,eta_bin=1.5):
    barrel_scale = 1./0.95
    endcap_scale = 1./0.85
    return np.where(eta<eta_bin,barrel_scale,endcap_scale)

def get_mu_correction(eta,eta_bin=2.1):
    barrel_scale = 1./0.98
    endcap_scale = 1./0.95
    return np.where(eta<eta_bin,barrel_scale,endcap_scale)


def apply_correction(jet_eta_bins,jet_pt_bins,sample,jet_pt_idx=0,jet_eta_idx=1):
    
    indecies_pt = np.searchsorted(jet_pt_bins, abs(sample[:,:,jet_pt_idx]),side='left')
    indecies_pt = np.where(indecies_pt==len(jet_pt_bins),len(jet_pt_bins)-1,indecies_pt)

    indecies_eta = np.searchsorted(jet_eta_bins, abs(sample[:,:,jet_eta_idx]),side='left')
    indecies_eta = np.where(indecies_eta==len(jet_eta_bins),len(jet_eta_bins)-1,indecies_eta)
    
    return indecies_pt,indecies_eta

def create_lorentz_particles(particles):
    return vector.array({"pt":particles[:,:,0], 
                                        "eta":particles[:,:,1],
                                        "phi":particles[:,:,2],
                                        "M":np.zeros(particles[:,:,2].shape)})

def get_met_ht(lorentz_all_particles,idx_jets=8):
    met = np.sqrt(np.sum(lorentz_all_particles.px, axis=1)**2 +
              np.sum(lorentz_all_particles.py, axis=1)**2) 
    metx = np.sum(lorentz_all_particles.px, axis=1) 
    mety = np.sum(lorentz_all_particles.py, axis=1)
    ht = np.sum(lorentz_all_particles[:,idx_jets:].pt,axis=1)
    return met,metx,mety,ht


def create_L1regression_data(data_file='',outfile_train='',outfile_test='',plot_dir='',jet_corr_dir='', jet_pt_filename='',jet_eta_filename='',train_test_split=0.8):
    with open(data_file, 'rb') as f:
        x_train, _,x_test, _, x_val, _,_,_,ids_train, ids_test, ids_val, ids_names  = pickle.load(f)

        data = np.concatenate((x_train,x_test,x_val),axis=0).astype('float32')
        ids = np.concatenate((ids_train,ids_test,ids_val),axis=0)
        del x_train,x_test,x_val, ids_train, ids_test,ids_val

    #removing Zll 
    data = data[ids!=2] #2 is index of Zll
    ids = ids[ids!=2] #2 is index of Zll

    idx_jet_0 = idx_particle_type['jet'][0]+1
    pt_idx = int(idx_feature['pt'])
    eta_idx = int(idx_feature['eta'])
    particles_in_jet = data[:,idx_jet_0:,:]
    other_particles = data[:,1:idx_jet_0,:]
    original_met = data[:,0,0]
    del data


    jet_pt_bins = [30,60,110,400]
    jes_correction_data = {}
    jes_correction_data_array = []
    jet_pt_filenames = ['%s_%d.csv'%(jet_pt_filename,jet_bin) for jet_bin in jet_pt_bins]
    for i,jet_bin in enumerate(jet_pt_bins) :
        jes_correction_data[jet_bin] = pd.read_csv(jet_corr_dir+'/'+jet_pt_filenames[i])
        jes_correction_data[jet_bin]['jet_response'] = 1./jes_correction_data[jet_bin]['jet_response']
        jes_correction_data_array.append(jes_correction_data[jet_bin][['abs_eta_bin_centers','jet_response']].values)
    jes_correction_data_array = np.stack(jes_correction_data_array, axis=0)

    fig, axs = plt.subplots(1,figsize=(10,8))
    for i,jet_bin in enumerate(jet_pt_bins) :
        _ = plt.plot(jes_correction_data[jet_bin]['abs_eta_bin_centers'],jes_correction_data[jet_bin]['jet_response'], label='Jet $p_T$ = %dGeV'%jet_bin,
                linestyle = 'None',
                marker=markers[i],
                color=colors[i])
    plt.legend(frameon=False)
    plt.ylabel('Correction factor')
    plt.xlabel('Jet |$\eta$|')
    plt.xlim(0, 4.75)
    plt.ylim(0.5, 1.25)
    plt.savefig(os.path.join(plot_dir, f'jet_energy_scale_correction.pdf'), bbox_inches="tight")
    plt.clf()

    jet_eta_bins = jes_correction_data_array[0,:,0]
    indecies_pt,indecies_eta = apply_correction(jet_eta_bins,jet_pt_bins,particles_in_jet,jet_pt_idx=pt_idx,jet_eta_idx=eta_idx)

    jer_eta_bins = ['0p5','3p2_4']
    jer_correction_data = {}
    jet_eta_filenames = ['%s_%s.csv'%(jet_eta_filename,jet_bin) for jet_bin in jer_eta_bins]
    for i,jet_bin in enumerate(jer_eta_bins) :
        jer_correction_data[jet_bin] = pd.read_csv(jet_corr_dir+'/'+jet_eta_filenames[i], names=['pt','jer'])

    popts, pcovs = [],[]
    for i,jet_bin in enumerate(jer_eta_bins) :
        popt, pcov = curve_fit(energy_res_function, xdata=jer_correction_data[jet_bin]['pt'].values, ydata=jer_correction_data[jet_bin]['jer'].values)
        popts.append(popt)
        pcovs.append(pcov)
        with open(jet_corr_dir+'fit_result_'+jet_eta_filenames[i].replace('.csv','.npy'), 'wb') as f:
            np.save(f, popts[i])
            np.save(f, pcovs[i])    
            
    fig, axs = plt.subplots(1,figsize=(10,8))            
    for i,jet_bin in enumerate(jer_eta_bins) :
        _ = plt.plot(jer_correction_data[jet_bin]['pt'],jer_correction_data[jet_bin]['jer'], label='Jet |$\eta$|=%s'%jet_bin,
                linestyle = 'None',marker=markers[i],color=colors[i])
        plt.plot(jer_correction_data[jet_bin]['pt'], energy_res_function(jer_correction_data[jet_bin]['pt'], *popts[i]),linewidth=3,
                      linestyle='--',color=colors[i],label='fit: a=%5.3f, b=%5.3f,c=%5.3f' % tuple(popts[i]))

    plt.legend(frameon=False)
    plt.ylabel('JER')
    plt.xlabel('Jet $p_T$')
    plt.semilogx()
    plt.xlim(0., 4000)
    plt.ylim(0., 0.5)
    plt.savefig(os.path.join(plot_dir, f'jet_energy_resolution.pdf'), bbox_inches="tight")
    plt.clf()

    jer_factor =  get_jer(particles_in_jet[:,:,pt_idx],particles_in_jet[:,:,eta_idx],popts)

    jet_pt_correction = jes_correction_data_array[indecies_pt,indecies_eta,1]
    jet_pt_jes_jer_factor = jet_pt_correction*\
                        rng.normal(1.0, scale=jer_factor*2, size=jet_pt_correction.shape)*\
                        rng.normal(1.1, scale=0.05, size=jet_pt_correction.shape)
    particles_in_jet_smeared = copy.deepcopy(particles_in_jet)
    particles_in_jet_smeared[:,:,pt_idx]*=jet_pt_jes_jer_factor

    res_mu = 0.05
    res_em = 0.2
    mu_corr_factor = get_mu_correction(other_particles[:,:idx_particle_type['mu'][1]+1,eta_idx:eta_idx+1])*\
                rng.normal(1.0, scale=res_mu, size=other_particles[:,:idx_particle_type['mu'][1]+1,eta_idx:eta_idx+1].shape)
    em_corr_factor = get_em_correction(other_particles[:,idx_particle_type['eg'][0]:idx_particle_type['eg'][1]+1,eta_idx:eta_idx+1])*\
                rng.normal(1.0, scale=res_em, size=other_particles[:,idx_particle_type['eg'][0]:idx_particle_type['eg'][1]+1,eta_idx:eta_idx+1].shape)
    other_particles_smeared = copy.deepcopy(other_particles)
    other_particles_smeared[:,:idx_particle_type['mu'][1]+1,pt_idx:pt_idx+1]*=mu_corr_factor
    other_particles_smeared[:,idx_particle_type['eg'][0]:idx_particle_type['eg'][1]+1,pt_idx:pt_idx+1]*=em_corr_factor


    #min true pt = 15 GeV for a jet, apply min pt cut of 20 GeV on smeared
    pt_cut = np.where(particles_in_jet_smeared[:,:,pt_idx]!=0,particles_in_jet_smeared[:,:,pt_idx]>=20,True)
    em_cut = np.where(other_particles_smeared[:,idx_particle_type['mu'][0],pt_idx]!=0,other_particles_smeared[:,idx_particle_type['mu'][0],pt_idx]>=23,True)
    mu_cut = np.where(other_particles_smeared[:,idx_particle_type['eg'][0],pt_idx]!=0,other_particles_smeared[:,idx_particle_type['eg'][0],pt_idx]>=23,True)
    select_reco_jets = np.where(
                        (np.all(pt_cut,axis=1)==True)[:,0] & 
                        ((np.all(em_cut,axis=1)==True) |
                        (np.all(mu_cut,axis=1)==True))
                    )[0] #all jets in the event have smeared pt > 20 GeV


    all_particles_smeared_selected = np.concatenate((other_particles_smeared[select_reco_jets],particles_in_jet_smeared[select_reco_jets]),axis=1)
    all_particles_selected = np.concatenate((other_particles[select_reco_jets],particles_in_jet[select_reco_jets]),axis=1)
    lorentz_all_particles_smeared = create_lorentz_particles(all_particles_smeared_selected)
    lorentz_all_particles = create_lorentz_particles(all_particles_selected)
    met_smeared,metx_smeared,mety_smeared,ht_smeared = get_met_ht(lorentz_all_particles_smeared)
    met,metx,mety,ht = get_met_ht(lorentz_all_particles)
    

    #saving created dataset 
    smeared_data = np.squeeze(np.stack((lorentz_all_particles_smeared.pt,
            lorentz_all_particles_smeared.eta,
            lorentz_all_particles_smeared.phi),axis=2),axis=-1)
    true_data = np.squeeze(np.stack((lorentz_all_particles.pt,
            lorentz_all_particles.eta,
            lorentz_all_particles.phi),axis=2),axis=-1)

    #split into training and testing samples :
    indecies_to_train = rng.choice(smeared_data.shape[0], int(train_test_split*smeared_data.shape[0]),replace=False)
    indecies_to_test = np.ones(smeared_data.shape[0], dtype=bool)
    indecies_to_test[indecies_to_train] = False  
                                         
    for name,indecies,output_file in zip(['train','test'],[indecies_to_train,indecies_to_test],[outfile_train,outfile_test]):
        with h5py.File(output_file, 'w') as h5f:        
            h5f.create_dataset('smeared_data', data=smeared_data[indecies])
            h5f.create_dataset('smeared_met', data=met_smeared[indecies].reshape(-1))
            h5f.create_dataset('smeared_ht', data=ht_smeared[indecies].reshape(-1))
            h5f.create_dataset('true_data', data=true_data[indecies])    
            h5f.create_dataset('true_met', data=met[indecies].reshape(-1))
            h5f.create_dataset('true_ht', data=ht[indecies].reshape(-1))   
            h5f.create_dataset('original_met', data=original_met[select_reco_jets][indecies].reshape(-1))             
            h5f.create_dataset('ids', data=ids[select_reco_jets][indecies].reshape(-1))    
            h5f.create_dataset('ids_names', data=ids_names[[0,1,3]])     #without Z


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Where is the data')
    parser.add_argument('--outfile_train', type=str, help='Where to save train data')
    parser.add_argument('--outfile_test', type=str, help='Where to save save data')
    parser.add_argument('--plot_dir', type=str, help='Plotting dir')
    parser.add_argument('--jet_pt_filename', type=str, help='Pt correction filename')
    parser.add_argument('--jet_eta_filename', type=str, help='Eta correction filename')
    parser.add_argument('--jet_corr_dir', type=str, help='Path to directory with jet energy and resolution correction files')
    parser.add_argument('--train_test_split', type=float, default=0.8, help='Train/test split for the final datasets')

    args = parser.parse_args()
    create_L1regression_data(**vars(args))



    