import h5py
import numpy as np
import argparse
import os
import shutil

idx_jet_0=8
idx_mu_0,idx_mu_1= 0,3
idx_eg_0,idx_eg_1= 4,7
pt_idx,eta_idx,phi_idx = 0,1,2

idx_particle_type = {'mu':[0,3],'eg':[4,7],'jet':[8,17]}
idx_feature = {'pt':0,'eta':1,'phi':2,'met':3,'ht':4,'pid':5}
code_particle_type = {'mu':1,'eg':2,'jet':3}

def preprocess_adjacency(A):
    A = A+np.eye(A.shape[1])
    D = np.array(np.sum(A, axis=2), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.nan_to_num(np.power(D,-0.5), posinf=0, neginf=0) # normalize (**-(1/2))
    D = np.asarray([np.diagflat(dd) for dd in D]) # and diagonalize
    return np.matmul(D, np.matmul(A, D))

def make_adjacencies(particles):
    real_p_mask = particles[:,:,0] > 0 # construct mask for real particles
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies


class MetResolutionEvaluator:
    def __init__(reco_met,true_met,process_ids,dnn_correction,mask=None):
        if mask is None:
            mask = np.ones(reco_met.shape[0],dtype=bool)
        self.reco_met = reco_met[mask]
        self.true_met = true_met[mask]
        self.process_ids = process_ids[mask]
        self.dnn_correction = dnn_correction[mask]
        self.compute_properties()

    def compute_properties(self):
        self.corr_met = self.reco_met*self.dnn_correction
        self.met_rel_diff = np.where(self.true_met>0,(self.true_met-self.reco_met)/self.true_met,0)
        self.met_rel_diff_corr = np.where(self.true_met>0,(self.true_met-self.corr_met)/self.true_met,0)
        self.met_ratio = np.where(self.true_met>0,self.true_met/self.reco_met,0)
        self.met_ratio_corr = np.where(self.true_met>0,self.true_met/self.corr_met,0)



    
class GraphCreator:
    def __init__(self, reco_data,reco_met,reco_ht, true_met,true_ht,process_ids,log_features=[]):
        self.reco_data = reco_data
        self.reco_met = reco_met
        self.reco_ht = reco_ht
        self.true_met = true_met
        self.true_ht = true_ht
        self.process_ids = process_ids
        self.log_features_idx = [idx_feature[f] for f in log_features]

        self.prepare_graph_features()
        self.prepare_adjacency()
        self.prepare_true_labels()


    def prepare_graph_features(self):
        graph_features = self.reco_data
        num_samples = graph_features.shape[0]
        num_mu = idx_particle_type['mu'][1]-idx_particle_type['mu'][0]+1
        num_eg = idx_particle_type['eg'][1]-idx_particle_type['eg'][0]+1
        num_jet = idx_particle_type['jet'][1]-idx_particle_type['jet'][0]+1

        #add met and ht to the inputs
        graph_features=np.concatenate((graph_features,
                        np.repeat(np.expand_dims(np.expand_dims(self.reco_met,axis=1),axis=1),graph_features.shape[1],axis=1),
                        np.repeat(np.expand_dims(np.expand_dims(self.reco_ht,axis=1),axis=1),graph_features.shape[1],axis=1)),
                        axis=-1)
        #add pid to the inputs
        graph_features=np.concatenate((graph_features,
                        np.concatenate((np.ones((num_samples,num_mu,1)),np.ones((num_samples,num_eg,1))*code_particle_type['eg'],np.ones((num_samples,num_jet,1))*code_particle_type['jet']),axis=1)),
                             axis=2)

        if len(self.log_features_idx)>0:
            graph_features[:,:,self.log_features_idx] = np.log(graph_features[:,:,self.log_features_idx]+1)

        self.features = graph_features

    def prepare_adjacency(self):
        A = make_adjacencies(self.features)
        A_tilde = preprocess_adjacency(A)
        self.adjacency =  A_tilde

    def prepare_true_labels(self):
        self.labels =  self.true_met/self.reco_met

    def apply_mask_on_graph(self,mask):
        return self.features[mask], self.adjacency[mask], self.labels[mask]





