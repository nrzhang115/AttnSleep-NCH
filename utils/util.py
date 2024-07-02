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
from imblearn.over_sampling import SMOTE

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
    # Reshape data from (samples, time_steps, features) to (samples, time_steps*features)
    # data is in the shape (num_samples, 3000, 1)
    nsamples, nx, ny = data.shape
    data_2d = data.reshape((nsamples, nx*ny))
    
    smote = SMOTE()
    data_resampled, labels_resampled = smote.fit_resample(data_2d, labels)
    
    # Reshape data back to (samples, time_steps, features)
    data_resampled = data_resampled.reshape((-1, nx, ny))
    
    # Debugging output to verify class distribution
    unique, counts = np.unique(labels_resampled, return_counts=True)
    print("Class distribution after SMOTE:", dict(zip(unique, counts)))
    
    return data_resampled, labels_resampled
############################################################################

def load_folds_data(np_data_path, n_folds):
    # Loads all .npz files from the specified directory
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    
    folds_data = {}
    
    file_to_use = files[0]
    
    # Verify the file path and its contents just before loading
    if not os.path.exists(file_to_use):
        print("Error: File does not exist", file_to_use)
    else:
        print("Loading original data from:", file_to_use)
        
    try:
    
        # Determine split indices for training and testing
        total_samples = len(np.load(file_to_use)['x'])
        print(f"Total labels in the files: {total_samples}")
        train_samples = int(0.7 * total_samples)
        test_samples = total_samples - train_samples
        
        # Split data indices
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        train_indices = indices[:train_samples]
        test_indices = indices[train_samples:]
        
        
        # Create training and testing data and labels
        train_data = np.load(file_to_use)['x'][train_indices]
        train_labels = np.load(file_to_use)['y'][train_indices]
        test_data = np.load(file_to_use)['x'][test_indices]
        test_labels = np.load(file_to_use)['y'][test_indices]
        print(f"Training set before oversampling: {train_data.shape}")
        print(f"Testing set before oversampling: {test_data.shape}")
        # print(f"Training set shape: {train_data.shape}")
        # print(f"Testing set data shape: {test_data.shape}")
        
        for fold_id in range(n_folds):
            # Perform oversampling on training data
            train_data, train_labels = oversample_data(train_data, train_labels)
            # Debugging output after oversampling
            print(f"Fold {fold_id} oversampled train_data shape: {train_data.shape}")
            
            # Save data to new file paths
            train_file_path = os.path.join(np_data_path, "train_data.npz")
            test_file_path = os.path.join(np_data_path, "test_data.npz")
            
            np.savez(train_file_path, x=train_data, y=train_labels)
            np.savez(test_file_path, x=test_data, y=test_labels)
            
            print(f"Train file path: {train_file_path}")
            print(f"Test file path: {test_file_path}")
            
            train_test_paths = [train_file_path,test_file_path]
            print(f"Folds data: {train_test_paths}")
            
            folds_data[fold_id] = [train_file_path, test_file_path]
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return folds_data


def calc_class_weight(labels_count):
    # Already applied oversampling
    num_classes = len(labels_count)
    class_weight = [1.0] * num_classes

    # Print current class distribution
    print(f"Class distribution: {labels_count}")
    print(f"Number of Classes: {num_classes}")

    # Calculate total instances for normalization
    total = sum(labels_count)
    
    # # Calculate weights inversely proportional to class frequencies
    # for i in range(num_classes):
    #     class_weight[i] = total / (num_classes * labels_count[i])

    # Adjust weights manually to emphasize importance of classes 0-5 more
    adjustment_factor = [5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.2]  # Increase for first 6, decrease for class 6
    class_weight = [w * adj for w, adj in zip(class_weight, adjustment_factor)]
    class_weight = [float(w * adj) for w, adj in zip(class_weight, adjustment_factor)]
    print(f"Calculated class weights: {class_weight}")

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
