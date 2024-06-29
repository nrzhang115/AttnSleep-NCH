import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
from scipy.signal import resample

def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"utils/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data

####################################################################
#  Oversampling to handle data imbalance 
def oversample_data(data, labels):
    unique, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)
    
    oversampled_data = []
    oversampled_labels = []
    
    for label in unique:
        class_data = data[labels == label]
        num_to_add = max_count - len(class_data)
        
        if num_to_add == 0:
            oversampled_data.append(class_data)
            oversampled_labels.extend([label] * len(class_data))
        else:
            oversampled_class_data = np.tile(class_data, (num_to_add // len(class_data) + 1, 1, 1))
            oversampled_class_data = oversampled_class_data[:num_to_add]
            
            oversampled_data.append(np.concatenate((class_data, oversampled_class_data), axis=0))
            oversampled_labels.extend([label] * (len(class_data) + num_to_add))
    
    oversampled_data = np.concatenate(oversampled_data, axis=0)
    oversampled_labels = np.array(oversampled_labels)
    
    indices = np.arange(len(oversampled_labels))
    np.random.shuffle(indices)
    # Debugging output to verify class distribution
    print("Class distribution after oversampling:", np.bincount(oversampled_labels))
    
    return oversampled_data[indices], oversampled_labels[indices]
############################################################################

def load_folds_data(np_data_path, n_folds):
    # Loads all .npz files from the specified directory
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))

    files_dict = dict()

    files_pairs = [[files[0],files[10],files[11],files[12]], [files[13],files[14],files[15],files[16]]]
    file_pair = []
    '''for i in range(n_folds * 32):
        file_pair.append(files[i])
        if (i + 1) % 32 == 0:
            files_pairs.append(file_pair)
            file_pair = []'''

    '''for key in files_dict:
        files_pairs.append(files_dict[key])'''
    files_pairs = np.array(files_pairs)
    print(files_pairs)
    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        # Determine training files by excluding subject_files
        training_files = list(set(files_pairs2) - set(subject_files))
        
        # Debugging prints
        print(f"Fold {fold_id + 1} - Files pairs:")
        print(subject_files)
        print(training_files)
        print(f"Training files count: {len(training_files)}")
        print(f"Testing/Validation files count: {len(subject_files)}")
        print(f"Total files: {len(training_files) + len(subject_files)}")
        print("=" * 40)
        
        folds_data[fold_id] = [training_files, subject_files]
    ###########################################################################
    # Load data from .npz files and apply oversampling
    def load_data_from_files(files):
        data_list = []
        labels_list = []
        for file in files:
            with np.load(file) as npzfile:
                data_list.append(npzfile['x'])
                labels_list.append(npzfile['y'])
        data = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return data, labels

    for fold_id in folds_data:
        train_files, test_files = folds_data[fold_id]
        # Debugging output before loading files
        print(f"Fold {fold_id} train_files: {train_files}")
        print(f"Fold {fold_id} test_files: {test_files}")
        
        train_data, train_labels = load_data_from_files(train_files)
        test_data, test_labels = load_data_from_files(test_files)

        # Oversample training data
        train_data, train_labels = oversample_data(train_data, train_labels)
        
        # Debugging output after oversampling
        print(f"Fold {fold_id} oversampled train_data shape: {train_data.shape}")
        print(f"Fold {fold_id} oversampled train_labels shape: {train_labels.shape}")

        folds_data[fold_id] = [train_files, test_files]
    #####################################################################################
    
    return folds_data


def calc_class_weight(labels_count):
    # Already applied oversampling 
    num_classes = len(labels_count)
    class_weight = [1.0] * num_classes
    print(f"Number of Classes: {num_classes}")
    
    # # Without Oversampling
    # total = np.sum(labels_count)
    # class_weight = dict()
    # num_classes = len(labels_count)
    # # Debugging information
    # print(f"Total: {total}")
    # print(f"Number of Classes: {num_classes}")
    # print(f"Labels Count: {labels_count}")

    # #############################################################################
    # # Appoarch 1 (From the original code)
    # factor = 1 / (num_classes)
    # # mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY
    # # Apporach 1 Modification Starts. 
    # # Adjust the class weight to address class imbalance.
    # # mu = [factor * 0.8, factor * 2.5, factor * 3.5, factor * 3.0, factor * 1.7, factor*4, factor*0.1]
    # mu = [factor] * num_classes
    # # Apporach 1 Modification Ends
    
    # # Debug Info
    # print(f"Mu: {mu}")
    
    # for key in range(num_classes):
    #     score = math.log(mu[key] * total / float(labels_count[key]))
    #     class_weight[key] = score if score > 1.0 else 1.0
    #     class_weight[key] = round(class_weight[key] * mu[key], 2)

    # class_weight = [class_weight[i] for i in range(num_classes)]
    
    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
