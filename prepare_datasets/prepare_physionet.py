'''
https://github.com/akaraspt/deepsleepnet
Copyright 2017 Akara Supratak and Hao Dong.  All rights reserved.
'''

import argparse
import glob
import math
import ntpath
import os
import shutil


from datetime import datetime

import numpy as np
import pandas as pd

from mne.io import concatenate_raws, read_raw_edf
import sleep_study as ss

# Label values
OD = 0,
OSA = 1
Hypopnea = 2
CSA = 3
OSH = 4
UNKNOWN = 5

stage_dict = {
    "Oxygen Desaturation": OD,
    "Obstructive Apnea": OSA,
    "Hypopnea": Hypopnea,
    "Central Apnea": CSA,
    "Obstructive Hypopnea": OSH,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "Oxygen Desaturation",
    1: "Obstructive Apnea",
    2: "Hypopnea",
    3: "Central Apnea",
    4: "Obstructive Hypopnea",
    5: "UNKNOWN"
}

ann2label = {
    "Oxygen Desaturation": 0,
    "Obstructive Apnea": 1,
    "Hypopnea": 2,
    "Central Apnea": 3,
    "Obstructive Hypopnea": 4,
    "Sleep stage R": 5,
    "Sleep stage W": 5,
    "Sleep stage 1": 5,
    "Sleep stage 2": 5,
    "Sleep stage 3": 5,
    "Sleep stage 4": 5,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH",
                        help="File path to the PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/AttnSleep_data/prepare_datasets/raw_eeg/C4-M1",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG C4-M1",
                        help="The selected channel")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*.tsv"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    ss.init()
    print('total number of sleep study files available:', len(ss.data.study_list))
    age_groups = list(zip(range(0, 18), range(1, 19))) + [(18, 100)]

    tmp = np.load('/srv/scratch/z5298768/AttnSleep_data/prepare_datasets/study_lists.npz', allow_pickle=True)
    study_lists = tmp["study_lists"]  # filenames that are in each age group
    num_segments = tmp["num_segments"]
    all_labels = tmp["all_labels"]

    # Change the code below according to the changes made in data.py regarding get_raw_eeg_and_labels
    for i, study_list in enumerate(study_lists):
        all_data = []
        all_labels = []
        for j, name in enumerate(study_lists[i]):
            data, labels = ss.data.get_raw_eeg_and_labels(name, args.data_dir, select_ch)
            
            # Modified: Skip the file if no specified events are found
            if len(labels) == 0:
                print(f"No specified events found in file: {name}. Skipping this file.")
                continue  # Skip this file if no specified events are found
            
            print(name)
            
            # Check if data and labels are consistent in shape
            if len(data) > 0 and data.shape[1:] == all_data.shape[1:]:
                all_data.extend(data)
                all_labels.extend(labels)
            else:
                print(f"Inconsistent data shape in file: {name}. Skipping this file.")
                continue

        if len(all_data) > 0:
            x = np.asarray(all_data).astype(np.float32)
            y = np.asarray(all_labels).astype(np.int32)
            # Save
            # filename = '/srv/scratch/z5298768/AttnSleep_data/prepare_datasets/raw_eeg/C4-M1/' + str(
            #     age_groups[i][0]) + '_' + str(age_groups[i][1]) + 'yrs_' + \
            #     datetime.now().isoformat(timespec='minutes') + '.npz'
            filename = os.path.join(args.output_dir, f'{age_groups[i][0]}_{age_groups[i][1]}yrs_{datetime.now().isoformat(timespec="minutes")}.npz')

            save_dict = {
                "x": x,
                "y": y,
                # "fs": sampling_rate,
                "ch_label": select_ch,
                # "header_raw": h_raw,
                # "header_annotation": h_ann,
            }
            np.savez(filename, **save_dict)
            print('features from', age_groups[i][0], 'to', age_groups[i][1], 'y.o. pts saved in', filename)
            print(' ')
            print("\n=======================================\n")
        else:
            print(f"No features to save for age group {age_groups[i][0]}-{age_groups[i][1]} years. Skipping.")


if __name__ == "__main__":
    main()
