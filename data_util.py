import numpy as np, os, sys
import pdb

def load_challenge_data_with_label(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    return data[:,:-1],data[:,-1]
    
def remove_nans(data):
    for col_idx in range(np.shape(data)[1]):
        nan_bools = np.isnan(data[:,col_idx])
        if np.all(nan_bools):
            return []
        else:
            nan_idcs = np.where(nan_bools)[0]
            good_idcs = np.where(nan_bools==False)[0]
            for idx in nan_idcs:
                if idx==0:
                    using_idx = np.min(good_idcs[good_idcs>idx])
                elif np.size(good_idcs[good_idcs<idx])==0:
                    using_idx = np.min(good_idcs[good_idcs>idx])
                else:
                    using_idx = np.max(good_idcs[good_idcs<idx])
                data[idx,col_idx] = data[using_idx, col_idx]
    return data
    
def add_time_relative_data(data):
    shape = np.shape(data)
    new_data = np.zeros((shape[0],shape[1]*5))
    for col_idx in range(shape[1]):
        new_data[:,col_idx] = data[:,col_idx]
        new_data[:,col_idx+shape[1]] = np.hstack((data[:1,col_idx],data[:-1,col_idx]))
        new_data[:,col_idx+(shape[1]*2)] = np.hstack((data[:2,col_idx],data[:-2,col_idx]))
        new_data[:,col_idx+(shape[1]*3)] = data[:,col_idx] - np.hstack((data[:1,col_idx],data[:-1,col_idx]))
        new_data[:,col_idx+(shape[1]*4)] = data[:,col_idx] - np.hstack((data[:2,col_idx],data[:-2,col_idx]))
    return new_data
    
    
def get_sepsis_labels(individual_label):
    sep_lbls = np.argwhere(individual_label==1)
    if np.size(sep_lbls) == 0:
        return individual_label
    t_opt = sep_lbls[0][0]
    shape = np.shape(individual_label)
    new_label = np.zeros(shape)
    curr_val = 1
    for i in range(t_opt-6,shape[0]):
        curr_val = min(curr_val,15)
        if i<0:
            curr_val += 1
        else:
            new_label[i] = curr_val
            curr_val+=1
    return new_label