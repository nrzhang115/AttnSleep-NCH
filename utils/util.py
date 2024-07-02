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
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    folds_data = {}

    if not files:
        raise FileNotFoundError("No .npz files found in the specified directory.")

    # Load first file to get shapes and labels
    file_to_use = np.load(files[0])
    x, y = file_to_use['x'], file_to_use['y']

    # Determine the number of instances per class and the minimum count to use for undersampling
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = np.min(counts)
    
    print("Class distribution before undersampling:", dict(zip(unique_classes, counts)))
    print("Minimum class size for undersampling:", min_count)

    # Create indices for undersampling
    undersampled_indices = np.hstack([np.random.choice(np.where(y == label)[0], min_count, replace=False)
                                      for label in unique_classes])

    # Shuffle the undersampled indices to mix classes
    np.random.shuffle(undersampled_indices)
    x = x[undersampled_indices]
    y = y[undersampled_indices]
    
    total_samples = len(y)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Split data into folds
    folds_data = {}
    for fold_id in range(n_folds):
        fold_size = total_samples // n_folds
        start = fold_id * fold_size
        end = start + fold_size if fold_id < n_folds - 1 else total_samples

        # Split indices for the current fold
        fold_indices = indices[start:end]
        fold_x = x[fold_indices]
        fold_y = y[fold_indices]

        # Further split the current fold into training and testing sets
        train_samples = int(0.7 * len(fold_x))
        train_indices = fold_indices[:train_samples]
        test_indices = fold_indices[train_samples:]

        train_data = x[train_indices]
        train_labels = y[train_indices]
        test_data = x[test_indices]
        test_labels = y[test_indices]

        # Save data to .npz files
        train_file_path = os.path.join(np_data_path, f"train_data_fold{fold_id}.npz")
        test_file_path = os.path.join(np_data_path, f"test_data_fold{fold_id}.npz")
        
        np.savez(train_file_path, x=train_data, y=train_labels)
        np.savez(test_file_path, x=test_data, y=test_labels)

        folds_data[fold_id] = [train_file_path, test_file_path]
        
        print(f"Data for fold {fold_id} saved: Train at {train_file_path}, Test at {test_file_path}")
        print(f"Training set shape: {train_data.shape}, Testing set data shape: {test_data.shape}")
    
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

    # Calculate inverse proportion and normalize
    proportions = labels_count / total
    inverse_proportions = 1 / (proportions + 1e-8)  # Adding a small constant to avoid division by zero
    class_weight = inverse_proportions / np.sum(inverse_proportions) * num_classes
    
    # Apply manual adjustments based on empirical needs
    # adjustments = np.array([0.8, 2.5, 3.5, 3.0, 1.7, 4, 0.1])
    # mu = normalized_mu * adjustments
    
    # print(f"Mu: {mu}")
    
    # for key in range(num_classes):
    #     # Using log to amplify the effect of very low values
    #     score = math.log(mu[key] * total / float(labels_count[key]))
    #     class_weight[key] = score if score > 1.0 else 1.0
    #     # Multiplying by mu again to adjust the final weight based on mu
    #     class_weight[key] = round(class_weight[key] * mu[key], 2)

    # class_weight = np.array([class_weight[i] for i in range(num_classes)], dtype=np.float32)
    
    
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
