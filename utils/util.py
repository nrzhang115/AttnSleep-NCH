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
        print(f"Training set shape: {train_data.shape}")
        print(f"Testing set data shape: {test_data.shape}")
        
        for fold_id in range(n_folds):
            
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
    
    # Without Oversampling
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)
    # Debugging information
    print(f"Total: {total}")
    print(f"Number of Classes: {num_classes}")
    print(f"Labels Count: {labels_count}")

    # Without Oversampling 
    # Adjust the class weight to address class imbalance.
    factor = 1 / (num_classes)
    # mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] 
    
    # mu = [factor * 0.8, factor * 2.5, factor * 3.5, factor * 3.0, factor * 1.7, factor*4, factor*0.1]
    mu = [0.03, 0.89, 1.90, 1.69, 0.32, 4.00, 0.10]
    # Debug Info
    # print(f"Mu: {mu}")
    
    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]
    
    
    print(f"Class weight: {class_weight}")
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
