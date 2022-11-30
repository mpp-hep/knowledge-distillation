import h5py
import numpy as np
import argparse
import os
import shutil
import random
from functools import partial
import tensorflow
from tensorflow import keras
import tensorflow.keras.backend as K

fixed_seed = 2021
random.seed(fixed_seed)
array32 = partial(np.array, dtype=np.float32)

idx_jet_0=8
idx_mu_0,idx_mu_1= 4,7
idx_eg_0,idx_eg_1= 0,3
pt_idx,eta_idx,phi_idx = 0,1,2

idx_particle_type = {'mu':[0,3],'eg':[4,7],'jet':[8,17]}
idx_feature_for_met = {'pt':0,'eta':1,'phi':2,'met':3,'ht':4,'pid':5}
idx_feature_for_ht = {'pt':0,'eta':1,'phi':2,'ht':3}
idx_feature = {'pt':0,'eta':1,'phi':2}
code_particle_type = {'mu':1,'eg':2,'jet':3}

def get_process_id_dict(ids_names):
    ids_names_dict = {}
    for el in ids_names:
        name_arr = (el.decode()).split('_')
        ids_names_dict[name_arr[0]]=float(name_arr[1])
    return ids_names_dict

def preprocess_adjacency(A):
    A = A+np.eye(A.shape[1],dtype=np.float32)
    D = np.array(np.sum(A, axis=2), dtype=np.float32) # compute outdegree (= rowsum)
    D = np.nan_to_num(np.power(D,-0.5), posinf=0, neginf=0) # normalize (**-(1/2))
    D = np.asarray([np.diagflat(dd) for dd in D], dtype=np.float32) # and diagonalize
    return np.matmul(D, np.matmul(A, D,dtype=np.float32),dtype=np.float32)

def make_adjacencies(particles):
    real_p_mask = particles[:,:,0] > 0 # construct mask for real particles
    adjacencies = (real_p_mask[:,:,np.newaxis] * real_p_mask[:,np.newaxis,:]).astype('float32')
    return adjacencies


class ResolutionEvaluator:
    'Resolution Evaluator '
    def __init__(self,reco_data,true_data,process_ids,dnn_correction,mask=None):
        if mask is None:
            mask = np.ones(reco_data.shape[0],dtype=bool)
        self.reco_data = array32(reco_data[mask]).reshape(-1,)
        self.true_data = array32(true_data[mask]).reshape(-1,)
        self.process_ids = process_ids[mask].astype(int)
        self.dnn_correction = array32(dnn_correction[mask]).reshape(-1,)
        self.compute_properties()

    def compute_properties(self):
        self.corr_data = self.reco_data*self.dnn_correction
        self.rel_diff = np.where(self.true_data>0,(self.true_data-self.reco_data)/self.true_data,0)
        self.rel_diff_corr = np.where(self.true_data>0,(self.true_data-self.corr_data)/self.true_data,0)
        self.tor = np.where(self.reco_data>0,self.true_data/self.reco_data,0)
        self.tor_corr = np.where(self.corr_data>0,self.true_data/self.corr_data,0)
        self.rot = np.where(self.true_data>0,self.reco_data/self.true_data,0)
        self.rot_corr = np.where(self.true_data>0,self.corr_data/self.true_data,0)


    
class METGraphCreator:
    'MET Graph Creator '
    def __init__(self, reco_data,reco_met,reco_ht, true_met,true_ht,process_ids,log_features=[]):
        mask = (reco_data[:,idx_jet_0,pt_idx]!=0).astype(bool) #mask events with 0 jets as we are not interested in this
        self.reco_data = array32(reco_data[mask])
        self.reco_met = array32(reco_met[mask])
        self.reco_ht = array32(reco_ht[mask])
        self.true_met = array32(true_met[mask])
        self.true_ht = array32(true_ht[mask])
        self.process_ids = process_ids[mask].astype(int)
        self.log_features_idx = [idx_feature_for_met[f] for f in log_features]

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
                                        np.concatenate((np.ones((num_samples,num_mu,1),dtype=int)*code_particle_type['mu'],
                                        np.ones((num_samples,num_eg,1),dtype=int)*code_particle_type['eg'],
                                        np.ones((num_samples,num_jet,1),dtype=int)*code_particle_type['jet'])
                                    ,axis=1)),
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
        self.labels = np.concatenate([self.labels.reshape(-1,1),self.features[:,0,idx_feature_for_met['met']].reshape(-1,1)],axis=1)
        
    def apply_mask_on_graph(self,mask):
        return self.features[mask], self.adjacency[mask], self.labels[mask], self.process_ids[mask]

    

class HTGraphCreator:
    'HT Graph Creator '
    def __init__(self, reco_data,reco_ht,true_ht,process_ids,log_features=[]):
        mask = (reco_data[:,idx_jet_0,pt_idx]!=0).astype(bool) #mask events with 0 jets as we are not interested in this
        self.reco_data = array32(reco_data[:,idx_jet_0:,:][mask]) #dataset consists of jets only
        self.reco_ht = array32(reco_ht[mask])
        self.true_ht = array32(true_ht[mask])
        self.process_ids = process_ids[mask].astype(int)
        self.log_features_idx = [idx_feature_for_ht[f] for f in log_features]

        self.prepare_graph_features()
        self.prepare_adjacency()
        self.prepare_true_labels()


    def prepare_graph_features(self):
        graph_features = self.reco_data
        num_samples = graph_features.shape[0]
        num_jet = idx_particle_type['jet'][1]-idx_particle_type['jet'][0]+1

        #add reco ht to the inputs
        graph_features=np.concatenate((graph_features,
                        np.repeat(np.expand_dims(np.expand_dims(self.reco_ht,axis=1),axis=1),graph_features.shape[1],axis=1)),
                        axis=-1)

        if len(self.log_features_idx)>0:
            graph_features[:,:,self.log_features_idx] = np.log(graph_features[:,:,self.log_features_idx]+1)

        self.features = graph_features

    def prepare_adjacency(self):
        A = make_adjacencies(self.features)
        A_tilde = preprocess_adjacency(A)
        self.adjacency =  A_tilde

    def prepare_true_labels(self):
        self.labels =  self.true_ht/self.reco_ht
        self.labels = np.concatenate([self.labels.reshape(-1,1),self.features[:,0,idx_feature_for_ht['ht']].reshape(-1,1)],axis=1)

    def apply_mask_on_graph(self,mask):
        return self.features[mask], self.adjacency[mask], self.labels[mask], self.process_ids[mask]




class DataGenerator(keras.utils.Sequence):
    'Data Generator for keras '
    def __init__(self, features, adjacency,graph_conv_filters,labels, batch_size,shuffle=True):
        self.features, self.adjacency, self.graph_conv_filters = features, adjacency,graph_conv_filters
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.features))
        self.num_batches = len(self.indices) // self.batch_size
        self.last_batch = len(self.indices) % self.batch_size

    def __len__(self):
        return self.num_batches + int(self.last_batch>0)

    def __getitem__(self, idx):
        end = min(len(self.indices), (idx + 1)*self.batch_size)
        indices_to_yiled = self.indices[idx*self.batch_size:end]
        X_batch, y_batch =  (self.features[indices_to_yiled],self.adjacency[indices_to_yiled],K.constant(self.graph_conv_filters[indices_to_yiled])),self.labels[indices_to_yiled]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indices)


def make_gen_callable(_gen):
    def gen():
        for x,y in _gen:
            yield x,y
    return gen



