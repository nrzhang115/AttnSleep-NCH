import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math

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

    if "78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"

    else:
        r_p_path = r"utils/r_permute_20.npy"

    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print ("============== ERROR =================")


    files_dict = dict()

    files_pairs = [[files[0],files[10],files[11]], [files[12],files[13],files[14]]]
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
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]

    return folds_data


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)
    # Debugging information
    print(f"Total: {total}")
    print(f"Number of Classes: {num_classes}")
    print(f"Labels Count: {labels_count}")

    #############################################################################
    # Appoarch 1 (From the original code)
    # factor = 1 / (num_classes)
    #mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY
    # Apporach 1 Modification Starts. Adjust the class weight to address class imbalance.
    # mu = [factor * 0.5, factor * 3, factor * 5, factor * 4, factor * 2]
    # Apporach 1 Modification Ends
    
    # Debug Info
    #print(f"Mu: {mu}")
    
    #for key in range(num_classes):
        #score = math.log(mu[key] * total / float(labels_count[key]))
        #class_weight[key] = score if score > 1.0 else 1.0
        #class_weight[key] = round(class_weight[key] * mu[key], 2)

    #class_weight = [class_weight[i] for i in range(num_classes)]
    
    ##############################################################
    # Appoarch 2 Modification Starts. Using a more systematic method to caculate class weight
    # Initial weights using inverse square root of class frequencies
    initial_weights = [1 / np.sqrt(count) for count in labels_count]
    sum_weights = sum(initial_weights)
    normalized_weights = [weight / sum_weights * num_classes for weight in initial_weights]

    # Apply a smaller adjustment factor
    adjustment_factor = 0.5
    adjusted_weights = [weight * adjustment_factor + (1 - adjustment_factor) * (1 / num_classes) for weight in normalized_weights]

    # Debug Info
    print(f"Adjusted Weights: {adjusted_weights}")
    
    for key in range(num_classes):
        score = math.log(adjusted_weights[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]
    # Apporach 2 Modification Ends .     
    #######################################################################
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
