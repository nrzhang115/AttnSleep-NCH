import os
import pandas as pd
from time import time

import sleep_study as ss

def check_annotations(df):
    event_dict = {k.lower(): 0 for k in ss.info.EVENT_DICT.keys()}

    for x in df.description:
        try:
            event_dict[x.lower()] += 1
        except:
            pass

    return any(event_dict.values())


def create_dataset(output_dir='~/sleep_study_dataset', percentage=40):
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    broken = []
    total = len(ss.data.study_list)
    sample_size = int(total * percentage / 100)  # Calculate 10% of the total dataset
    sample_study_list = ss.data.study_list[:sample_size]

    for i, name in enumerate(sample_study_list):
        if i % 100 == 0:
            print('Processing %d of %d' % (i, sample_size))

        path = os.path.join(ss.data_dir, 'Sleep_Data', name + '.tsv')
        df = pd.read_csv(path, sep='\t')

        if not check_annotations(df):
            broken.append(name)

    print('Processed %d files' % i)
    print('%d files have no labeled sleeping stage' % len(broken))

    path = os.path.join(output_dir, 'Sleep_Data')
    os.makedirs(path, exist_ok=True)

    output_path = os.path.join(output_dir, 'Sleep_Data.tar.xz')
    cmd = 'cd %s && tar -cf - Health_Data | xz -T0 > %s' % (ss.data_dir, output_path)

    print('Compressing health data')
    start = time()
    os.system(cmd)
    end = time()
    print('Compressed, used %.2f seconds' % (end - start))

    for i, name in enumerate(sample_study_list):
        cmd = 'cd %s/Sleep_Data && tar -cf - %s* | xz -T0 > %s/Sleep_Data/%s.tar.xz' % (ss.data_dir, name, output_dir, name)
        print('Compressing %s.edf and %s.tsv, %d of %d' % (name, name, i + 1, sample_size))
        start = time()
        os.system(cmd)
        end = time()
        print('Compressed, used %.2f seconds' % (end - start))
    
    
    # Check if 10% of the data has been prepared
    prepared_files = os.listdir(os.path.join(output_dir, 'Sleep_Data'))
    prepared_files_count = len([f for f in prepared_files if f.endswith('.tar.xz')])
    expected_files_count = sample_size

    print(f"Expected number of prepared files: {expected_files_count}")
    print(f"Actual number of prepared files: {prepared_files_count}")

    if prepared_files_count == expected_files_count:
        print("The preparation of 10% of the dataset is verified.")
    else:
        print("The preparation of 10% of the dataset is not accurate.")


def get_studies_by_patient_age(low, high, txt_path='age_file.csv'):
    study_list = []
    ages = []
    df = pd.read_csv(txt_path, sep=",", dtype={'FILE_NAME': 'str', 'AGE_AT_SLEEP_STUDY_DAYS': 'int'})
    
    df = df[(df.AGE_AT_SLEEP_STUDY_DAYS >= low*365) & (df.AGE_AT_SLEEP_STUDY_DAYS < high*365)]
    print("found", len(df), "patients between", low, "(incl.) and", high, "(excl.) years old.")

    return df.FILE_NAME.tolist(), df.AGE_AT_SLEEP_STUDY_DAYS.tolist()

def annotation_stats(percentage=40):  # Default percentage to 10 for consistency
    output_dir = './'

    broken = []
    total = len(ss.data.study_list)
    sample_size = int(total * percentage / 100)  # Calculate the percentage of the total dataset
    sample_study_list = ss.data.study_list[:sample_size]

    for i, name in enumerate(sample_study_list):

        if i % 100 == 0:
            print('Processing %d of %d' % (i, sample_size))

        path = os.path.join(ss.data_dir, 'Sleep_Data', name + '.tsv')
        df = pd.read_csv(path, sep='\t')

        if not check_annotations(df):
            broken.append(name)

    print('Processed %d files' % i)
    print('%d files have no labeled sleeping stage' % len(broken))

    # Check if the correct percentage of data is processed
    expected_files_count = sample_size
    processed_files_count = len(sample_study_list) - len(broken)

    print(f"Expected number of processed files: {expected_files_count}")
    print(f"Actual number of processed files: {processed_files_count}")

    if processed_files_count == expected_files_count:
        print(f"The processing of {percentage}% of the dataset is verified.")
    else:
        print(f"The processing of {percentage}% of the dataset is not accurate.")

