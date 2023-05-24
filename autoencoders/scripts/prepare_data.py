import argparse
import h5py
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def check_proportions(dataset):
    for i, ids in enumerate(['W', 'QCD', 'Z', 'tt']):
        print(f'Percent of {ids} is {dataset[dataset==i].shape[0]/dataset.shape[0]*100:0.2f}')


def filter_no_leptons(data, background_ID=None):
    locs = np.logical_or(np.greater_equal(data[:,1,0], 23), np.greater_equal(data[:,5,0], 23))
    data_filtered = data[locs]

    if background_ID is not None:
        print(f'IDs before filtering:')
        check_proportions(background_ID)
        background_ID_filtered = background_ID[locs]

        print(f'IDs after filtering: ')
        check_proportions(background_ID_filtered)
        return data_filtered, background_ID_filtered

    return data_filtered


def remove_jets_mod_eta_more4(data):
    """
    Remove jets eta >4 or <-4 and fix pT ordering
    """
    data[:,9:19,0] = np.where(data[:,9:19,1]>4,0,data[:,9:19,0])
    data[:,9:19,0] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,0])
    data[:,9:19,2] = np.where(data[:,9:19,1]>4,0,data[:,9:19,2])
    data[:,9:19,2] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,2])
    data[:,9:19,1] = np.where(data[:,9:19,1]>4,0,data[:,9:19,1])
    data[:,9:19,1] = np.where(data[:,9:19,1]<-4,0,data[:,9:19,1])
    d_jets = data[:,9:19].copy()

    # order with highest particle pT first
    ind = np.argsort(-1 * d_jets[:,:,0])
    ind = np.stack(d_jets.shape[2]*[ind], axis=1)
    d_jets = np.take_along_axis(d_jets.transpose(0, 2, 1), ind, axis=2).transpose(0, 2, 1)
    data[:,9:19] = d_jets

    return data


def read_file(bsm_name, file_str):
    with h5py.File(file_str,'r') as h5f:
        dataset = np.array(h5f['Particles'][:,:,:-1])
    # remove jets eta >4 or <-4
    dataset = remove_jets_mod_eta_more4(dataset)
    n_before = dataset.shape[0]
    print(f'{bsm_name} dataset before filter',n_before,'after filter',dataset.shape[0],\
        'cut away',(n_before-dataset.shape[0])/n_before*100,r'%')
    dataset = dataset.reshape(dataset.shape[0],dataset.shape[1],dataset.shape[2],1)

    return dataset


def main(args):

    # read QCD data
    with h5py.File(args.input_file, 'r') as h5f:

        # remove last feature, which is the type of particle
        data = h5f['Particles'][:args.events,:,:-1]
        background_ID = h5f['y'][:args.events]
        background_ID_names = h5f['y_labels'][:]
        data, background_ID = shuffle(data, background_ID)

    # remove jets eta >4 or <-4
    data = remove_jets_mod_eta_more4(data)
    n_before = data.shape[0]
    data, background_ID = filter_no_leptons(data, background_ID)
    print('Background before filter',n_before,'after filter',data.shape[0],\
        'cut away',(n_before-data.shape[0])/n_before*100,r'%')

    # fit scaler to the full data
    pt_scaler = StandardScaler()
    data_target = np.copy(data)
    data_target[:,:,0] = pt_scaler.fit_transform(data_target[:,:,0])
    data_target[:,:,0] = np.multiply(data_target[:,:,0], np.not_equal(data[:,:,0],0))

    # define training, test and validation datasets
    x_train, x_test, y_train, y_test, background_ID_train, background_ID_test = \
        train_test_split(data, data_target, background_ID, test_size=0.2, shuffle=True)
    del data, data_target, background_ID
    x_train, x_val, y_train, y_val, background_ID_train, background_ID_val = \
        train_test_split(x_train, y_train, background_ID_train,
            test_size=0.25, shuffle=True)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)

    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], y_val.shape[2], 1)

    np.savez(args.output_file,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        )

    extra_bsm = dict()
    for bsm_name, bsm_file in args.input_bsm:
        if bsm_name=='leptoquark':
            leptoquark = read_file(bsm_name, bsm_file)
        elif bsm_name=='ato4l':
            ato4l = read_file(bsm_name, bsm_file)
        elif bsm_name=='hChToTauNu':
            hChToTauNu = read_file(bsm_name, bsm_file)
        elif bsm_name=='hToTauTau':
            hToTauTau = read_file(bsm_name, bsm_file)
        else:
            extra_bsm[bsm_name] = read_file(bsm_name, bsm_file)

    np.savez(args.bsm_output_file,
        leptoquark=leptoquark,
        ato4l=ato4l,
        hChToTauNu=hChToTauNu,
        hToTauTau=hToTauTau,
        **extra_bsm)

    with open(args.pt_scaler_file, 'wb') as handle:
        pickle.dump(pt_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    np.savez(args.background_ID_output_file,
        background_ID_train=background_ID_train,
        background_ID_test=background_ID_test,
        background_ID_val=background_ID_val,
        background_ID_names=background_ID_names,
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('input_file', type=str,
        help='Input file with the data')
    parser.add_argument('output_file', type=str,
        help='Output file to save training, validation and testing datasets')
    parser.add_argument('bsm_output_file', type=str,
        help='Output file to save BSM pre-processed data')
    parser.add_argument('pt_scaler_file', type=str,
        help='Output file to save pt scaler file')
    parser.add_argument('background_ID_output_file', type=str,
        help='Output file to save info about the background ID')

    parser.add_argument('--input-bsm', action='append', nargs='+',
        help='Input file for generated BSM')
    parser.add_argument('--events', type=int, default=-1,
        help='How many events to proceed')
    args = parser.parse_args()
    main(args)
