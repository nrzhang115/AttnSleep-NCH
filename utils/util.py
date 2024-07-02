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

# Perform undersample for class 6 (Majority class)
def undersample_indices(labels, target_class=6, undersample_factor=10):
    """Returns indices after undersampling the majority class."""
    unique, counts = np.unique(labels, return_counts=True)
    target_count_majority = min(counts) * undersample_factor  # Adjust factor as needed

    undersampled_indices = []
    for class_value in unique:
        class_indices = np.where(labels == class_value)[0]
        if class_value == target_class:
            np.random.shuffle(class_indices)
            class_indices = class_indices[:target_count_majority]  # Reduce to target count
        undersampled_indices.extend(class_indices)

    np.random.shuffle(undersampled_indices)  # Shuffle to mix classes well
    return undersampled_indices

def load_folds_data(np_data_path, n_folds):
    # Loads all .npz files from the specified directory
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    
    folds_data = {}
    
    file_to_use = files[0]
    
    data = np.load(file_to_use)['x']
    labels = np.load(file_to_use)['y']
    
    # Verify the file path and its contents just before loading
    if not os.path.exists(file_to_use):
        print("Error: File does not exist", file_to_use)
    else:
        print("Loading original data from:", file_to_use)
        
    try:
        
        
        # Use the undersampling function to get the right indices
        undersampled_indices = undersample_indices(labels)
        undersampled_data = data[undersampled_indices]
        undersampled_labels = labels[undersampled_indices]
        
        total_samples = len(undersampled_labels)
        print(f"Total labels in the files: {total_samples}")
        train_samples = int(0.7 * total_samples)
        test_samples = total_samples - train_samples
        
        # Split data indices
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        train_indices = indices[:train_samples]
        test_indices = indices[train_samples:]
        
        # Create training and testing data and labels
        train_data = undersampled_data[train_indices]
        train_labels = undersampled_labels[train_indices]
        test_data = undersampled_data[test_indices]
        test_labels = undersampled_labels[test_indices]
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

    # Calculate proportional inverse to guide the mu adjustment
    proportions = labels_count / total
    inverse_proportions = 1 / proportions
    normalized_mu = inverse_proportions / np.sum(inverse_proportions) * num_classes  # Normalize and scale by number of classes

    # Adjust mu based on specific needs
    mu = normalized_mu * np.array([0.5, 2.5, 3.0, 2.5, 1.5, 3.5, 1.0])

    # Debug Info
    print(f"Mu: {mu}")
    
    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]
    
    
    print(f"Mu: {class_weight}")
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
