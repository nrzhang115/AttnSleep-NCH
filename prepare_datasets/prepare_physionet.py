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

# APNEA = 0  # Binary training 
# OTHER = 1
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    # "APNEA": APNEA,
    # "OTHER": OTHER
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
    
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
    # 0: "APNEA",
    # 1: "OTHER"
}

ann2label = {
    # "Oxygen Desaturation": 0,
    # "Obstructive Apnea": 0,
    # "Hypopnea": 0,
    # "Central Apnea": 0,
    # "Obstructive Hypopnea": 0,
    # 'Mixed Apnea':  0, 
    # 'Sleep stage R': 1,
    # 'Sleep stage W': 1,
    # 'Sleep stage N1': 1,
    # 'Sleep stage N2': 1,
    # 'Sleep stage N3': 1,
    # 'EEG Arousal': 1,
    # 'Oximeter Event': 1,
    # 'Move': 1,
    # 'Limb Movement': 1,
    # 'SHORT_EVENT': 1
    
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5,
    "SHORT_EVENT": 5
}

EPOCH_SEC_SIZE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/srv/scratch/speechdata/sleep_data/NCH",
                        help="File path to the PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="/srv/scratch/z5298768/AttnSleep_data/prepare_datasets/raw_eeg_stages/C3-M2",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG C3-M2",
                        help="The selected channel")
    # Initialise 10% of the dataset
    parser.add_argument("--percentage", type=int, default=10, 
                    help="Percentage of the dataset to process")
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
    # Initialize study list with the given percentage
    total_studies = len(ss.data.study_list)
    sample_size = int(total_studies * args.percentage / 100)
    sample_study_list = ss.data.study_list[:sample_size]
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
            if name not in sample_study_list:
                continue
            
            data, labels = ss.data.get_raw_eeg_and_labels(name, args.data_dir, select_ch)
            
            # Modified: Skip the file if no specified events are found
            if len(labels) == 0:
                print(f"No specified events found in file: {name}. Skipping this file.")
                continue  # Skip this file if no specified events are found
            
            print(name)
            all_data.extend(data)
            all_labels.extend(labels)

        if len(all_data) > 0:
            x = np.asarray(all_data).astype(np.float32)
            y = np.asarray(all_labels).astype(np.int32)
            print(f"Final data shape to be saved: {x.shape}")
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
