import ast
import csv
import os
import pandas as pd
import argparse
from pickle import dump
import shutil
import random
from tqdm import tqdm

import numpy as np
from tfsnippet.utils import makedirs

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', type=str, default='Backblaze')
parser.add_argument('--output_folder', type=str, default='processed')
args = parser.parse_args()


def load_data(dataset_folder, output_folder):
    serial_number_dict = {}

    filelist = os.listdir(dataset_folder)
    fail_files = [filename for filename in filelist if filename.startswith('1')]
    no_fail_files = [filename for filename in filelist if filename.startswith('0')]
    train_files = fail_files[:int(len(fail_files) * 0.2)] + no_fail_files[:int(len(no_fail_files) * 0.8)]
    random.shuffle(train_files)
    test_files = fail_files[int(len(fail_files) * 0.2):] + no_fail_files[int(len(no_fail_files) * 0.8):]
    random.shuffle(test_files)

    def create_files(filelist, name):
        arr_list = []
        for filename in tqdm(filelist):
            # Load csv file with header
            tmp = pd.read_csv(os.path.join(dataset_folder, filename), header=0)
            # Get serial number
            serial_number = filename.split('_')[2]
            if serial_number not in serial_number_dict:
                serial_number_dict[serial_number] = len(serial_number_dict) + 1
            # Add serial number column
            tmp.insert(6, 'serial_number_num', serial_number_dict[serial_number])
            # Keep only certain columns
            meta_columns = tmp.columns.values.tolist()[:6] + ['serial_number_num']
            smart_columns = ['smart_1_normalized',
                             'smart_5_normalized', 'smart_5_raw',
                             'smart_7_normalized',
                             'smart_9_raw',
                             'smart_12_raw',
                             'smart_187_normalized', 'smart_187_raw',
                             'smart_193_normalized', 'smart_193_raw',
                             'smart_197_normalized', 'smart_197_raw',
                             'smart_198_normalized', 'smart_198_raw',
                             'smart_199_raw']
            tmp = tmp[meta_columns + smart_columns]
            # Place label column at the end
            tmp = tmp[[col for col in tmp.columns if col != 'failure'] + ['failure']]
            # Drop rows with missing values
            tmp = tmp.dropna()
            # If anomaly (at least one failure value is 1), don't use last 60 days before first failure
            if tmp['failure'].sum() > 0:
                first_anomaly_index = np.where(tmp['failure'].values == 1)[0][0]
                if name == 'train':
                    tmp = tmp.iloc[:first_anomaly_index - 60, :]
                elif name == 'test':
                    tmp = tmp.iloc[:first_anomaly_index + 1, :]
                    # Set 7 days before failure as anomaly
                    tmp.iloc[-7:, -1] = 1
            if tmp.shape[0] == 0:
                print("Empty file: " + filename)
                continue
            # assert that label is always 0
            if name == 'train':
                assert tmp['failure'].sum() == 0

            if name == 'test':
                # numpy array from dataframe starting from column 6
                arr = tmp.iloc[:, 5:].values
            elif name == 'train':
                # Exclude last column (label)
                arr = tmp.iloc[:, 5:-1].values
            # Add to list
            arr_list.append(arr)
        # Concatenate all arrays
        arr = np.concatenate(arr_list, axis=0)
        # Save to file
        with open(os.path.join(output_folder, name + ".pkl"), "wb") as file:
            dump(arr, file)

    create_files(train_files, 'train')
    create_files(test_files, 'test')
    # Save serial number dictionary
    with open(os.path.join(output_folder, "serial_number_dict.pkl"), "wb") as file:
        dump(serial_number_dict, file)


if __name__ == '__main__':
    # Delete output folder if exists
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    # Create output folder
    makedirs(args.output_folder, exist_ok=True)
    load_data(args.dataset_folder, args.output_folder)
