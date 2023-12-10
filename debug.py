import json

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from omni_anomaly.eval_methods import bf_search


def plot_histogram(test_score, test_label, name):

    # Concatenate test_data, test_labels and test_scores
    test_data = pd.concat([pd.DataFrame(test_label), pd.DataFrame(test_score)], axis=1)
    # Rename columns
    test_data.columns = ['label', name]

    # Split into anomalies and normal
    test_data_anomalies = test_data[test_data['label'] == 1]
    test_data_normal = test_data[test_data['label'] == 0]

    # Plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(test_data_anomalies[name], bins=[-float('inf')] + list(range(-100, 100, 1)) + [float('inf')], density=True,
            alpha=0.5, color='r')
    ax.hist(test_data_normal[name], bins=[-float('inf')] + list(range(-100, 100, 1)) + [float('inf')], density=True,
            alpha=0.5, color='b')
    ax.set_xlabel('Anomaly score')
    ax.set_ylabel('Density')
    ax.legend(['Anomalies', 'Normal'])
    ax.set_title('Histogram of anomaly scores for ' + name)
    plt.show()

def generate_bf_search_pkl(result_dir, test_score, y_test, min):
    # Generate bf_search.pkl
    max = 100
    while True:
        search = int((min + max) / 2)
        _, _, arr = bf_search(test_score, y_test,
                              start=search,
                              end=search + 1,
                              step_num=1,
                              display_freq=1)
        if arr[0, 2] > 0.01:
            max = search
        else:
            min = search
        if max - min <= 1:
            break
    t, th, arr = bf_search(test_score, y_test,
                           start=search,
                           end=100,
                           step_num=(100 - search),
                           display_freq=10)
    # Save array in result folder
    pickle.dump(arr, open(os.path.join(result_dir, 'bf_search.pkl'), 'wb'))


def get_bf_search_score(result_dir, test_data_path, name, thr=None, days=14):
    test_score = pickle.load(open(os.path.join(result_dir, 'test_score.pkl'), 'rb'))
    test_data = pickle.load(open(test_data_path, 'rb'))
    config = json.load(open(os.path.join(result_dir, 'config.defaults.json'), 'r'))
    window_length = config['window_length']
    # Convert test data to DataFrame
    test_df = pd.DataFrame(test_data)
    # Group test_data by first column (serial number) and cut off first window_length rows
    test_df = test_df.groupby(test_df.columns[0], sort=False).apply(lambda x: x.iloc[window_length - 1:, :])
    # Reset index
    test_df = test_df.reset_index(drop=True)
    # Group by serial number, if last row is anomaly, label as anomaly
    anomaly_count = test_df.groupby(test_df.columns[0], sort=False).apply(lambda x: x.iloc[-1, -1]).value_counts()
    print(name + ": Anomaly count: ", anomaly_count)
    def change_label(x):
        if x.iloc[-1, -1] == 1:
            x.iloc[:, -1] = 0
            x.iloc[:-days, -1] = 1
        return x
    test_df = test_df.groupby(test_df.columns[0], sort=False).apply(change_label)
    # Concatenate test_score
    test_df = pd.concat([test_df, pd.DataFrame(test_score)], axis=1)
    # Use last row as label
    test_label = test_df.iloc[:, -2].values

    # If bf_search.pkl does not exist, generate it
    if not os.path.exists(os.path.join(result_dir, 'bf_search.pkl')):
        generate_bf_search_pkl(result_dir, test_score, test_label, -100000)

    bf_search_score = pickle.load(open(os.path.join(result_dir, 'bf_search.pkl'), 'rb'))

    # Get FDR for FAR = 0.01
    fdr = bf_search_score[:, 1][np.where(bf_search_score[:, 2] > 0.01)][0]
    thr_001 = bf_search_score[:, -1][np.where(bf_search_score[:, 2] > 0.01)][0]
    print(name + ": FDR for FAR = 0.01: ", fdr, " with threshold: ", thr_001)
    if thr is not None:
        fdr = bf_search_score[:, 1][np.where(bf_search_score[:, -1] > thr)][0]
        print(name + ": FDR for FAR = 0.01: ", fdr, " with threshold: ", thr)

    plot_histogram(test_score, test_label, name)

    return bf_search_score, thr_001


if __name__ == '__main__':

    st8000, thr1 = get_bf_search_score('results/tuned_model_st8000/results', 'processed_st8000/test.pkl', 'st8000')
    st8000_2, _ = get_bf_search_score('results/tuned_model_st8000_2/results', 'processed_st8000/test.pkl', 'st8000')
    st8000_exp1, thr2 = get_bf_search_score('results/tuned_model_st8000_exp1/results', 'processed_st12000/test.pkl', 'st8000_exp1', thr1)

    fig, ax = plt.subplots(1, 1)

    ax.plot(st8000[:, 2], st8000[:, 1], label='st8000 pre-train')
    ax.plot(st8000_2[:, 2], st8000_2[:, 1], label='st8000 pre-train 2')
    ax.plot(st8000_exp1[:, 2], st8000_exp1[:, 1], label='st8000 TFL st12000 no train')
    # Vertical line at 0.01
    ax.axvline(x=0.01, color='r', linestyle='--')
    ax.legend()
    # Set x label
    ax.set_xlabel('FAR')
    # Set y label
    ax.set_ylabel('FDR')
    # Set title
    ax.set_title('FDR-FAR Tradeoff')
    # x lim
    ax.set_xlim([0, 0.3])
    plt.show()
