import ast
import csv
import os
import sys
import pandas as pd
import argparse
from pickle import dump
import shutil
import random

import numpy as np
from tfsnippet.utils import makedirs

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Backblaze')
parser.add_argument('--dataset_folder', type=str, default='Backblaze')
parser.add_argument('--output_folder', type=str, default='processed')
args = parser.parse_args()

makedirs(args.output_folder, exist_ok=True)


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset, dataset_folder, output_folder):
    serial_number_dict = {}
    if dataset == 'Backblaze':
        filelist = os.listdir(dataset_folder)
        random.shuffle(filelist)
        fail_files = [filename for filename in filelist if filename.startswith('1')]
        no_fail_files = [filename for filename in filelist if filename.startswith('0')]
        train_files = fail_files[:int(len(fail_files) * 0.5)] + no_fail_files[:int(len(no_fail_files) * 0.8)]
        random.shuffle(train_files)
        test_files = fail_files[int(len(fail_files) * 0.5):] + no_fail_files[int(len(no_fail_files) * 0.8):]
        random.shuffle(test_files)

        def create_files(filelist, name):
            for filename in filelist:
                print(dataset, name, filename)
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
                # Print index of rows with at least one missing value
                missing_index = np.where(tmp.isnull().any(axis=1))[0]
                if len(missing_index) > 0:
                    print(filename + " missing val in rows: " + str(missing_index) + "(total: " + str(len(missing_index)) + ")")
                # Drop rows with missing values
                tmp = tmp.dropna()
                # If it is an anomaly disk (last row is 1), don't use last 60 days for training
                if name == 'train' and tmp['failure'].values[-1] == 1:
                    tmp = tmp.iloc[:-60, :]
                if tmp.shape[0] == 0:
                    print("Empty file: " + filename)
                    continue
                # assert that label is always 0
                if name == 'train':
                    assert tmp['failure'].sum() == 0
                # numpy array from dataframe starting from column 7
                arr = tmp.iloc[:, 6:].values
                # Save to pickle file
                with open(os.path.join(output_folder, filename.strip('.csv') + "_" + name + ".pkl"), "wb") as file:
                    dump(arr, file)

                if name == 'test':
                    # Get label
                    label = tmp['failure'].values
                    # Set 7 days before failure as anomaly
                    if label[-1] == 1:
                        label[-7:] = 1
                    # Save to pickle file
                    with open(os.path.join(output_folder, filename.strip('.csv') + "_" + name + "_label.pkl"), "wb") as file:
                        dump(label, file)

        create_files(train_files, 'train')
        create_files(test_files, 'test')

    elif dataset == 'SMD':
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder)
    elif dataset == 'SMAP' or dataset == 'MSL':
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ['train', 'test']:
            concatenate_and_save(c)


if __name__ == '__main__':
    datasets = ['SMD', 'SMAP', 'MSL', 'Backblaze']
    if args.dataset not in datasets:
        sys.exit("Unknown dataset: " + args.dataset)
    # Delete output folder if exists
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    # Create output folder
    makedirs(args.output_folder, exist_ok=True)
    load_data(args.dataset, args.dataset_folder, args.output_folder)
