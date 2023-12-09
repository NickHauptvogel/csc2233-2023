import json

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from omni_anomaly.eval_methods import bf_search


def plot_histogram(test_score, test_label, name):
    # Load test_scores
    #test_scores = pickle.load(open(test_scores_path, 'rb'))
    # Load test_labels
    #test_labels = pickle.load(open(test_labels_path, 'rb'))
    #test_data = pickle.load(open(test_data_path, 'rb'))
    #test_data = pd.DataFrame(test_data)
    # Group test_data by first column (serial number) and cut off first 24 rows
    #test_data = test_data.groupby(test_data.columns[0], sort=False).apply(lambda x: x.iloc[24:, :])
    # Reset index
    #test_data = test_data.reset_index(drop=True)

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
    ax.set_yscale('log')
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


if __name__ == '__main__':

    result_dir = 'results/tuned_model_st8000/results'
    test_score = pickle.load(open(os.path.join(result_dir, 'test_score.pkl'), 'rb'))
    test_data = pickle.load(open('processed_st8000/test.pkl', 'rb'))
    config = json.load(open(os.path.join(result_dir, 'config.defaults.json'), 'r'))
    window_length = config['window_length']
    # Convert test data to DataFrame
    test_df = pd.DataFrame(test_data)
    # Group test_data by first column (serial number) and cut off first window_length rows
    test_df = test_df.groupby(test_df.columns[0], sort=False).apply(lambda x: x.iloc[window_length - 1:, :])
    # Reset index
    test_df = test_df.reset_index(drop=True)
    test_df = test_df.groupby(test_df.columns[0], sort=False).apply(lambda x: x.iloc[window_length - 1:, :])
    test_df = test_df.reset_index(drop=True)
    # Use last row as label
    test_label = test_df.iloc[:, -1].values
    test_data = test_df.iloc[:, :-1].values

    # If bf_search.pkl does not exist, generate it
    if not os.path.exists(os.path.join(result_dir, 'bf_search.pkl')):
        generate_bf_search_pkl(result_dir, test_score, test_label, -100000)

    st8000 = pickle.load(open(os.path.join(result_dir, 'bf_search.pkl'), 'rb'))
    st8000_st12000 = pickle.load(open('results/model_test_st8000_st12000/results/bf_search.pkl', 'rb'))
    st8000_st12000_2m = pickle.load(open('results/model_test_st8000_st12000_small/results/bf_search.pkl', 'rb'))

    # Get FDR for FAR = 0.01
    fdr = st8000[:, 1][np.where(st8000[:, 2] > 0.01)][0]
    thr_pretrain = st8000[:, -1][np.where(st8000[:, 2] > 0.01)][0]
    print("ST8000: FDR for FAR = 0.01: ", fdr, " with threshold: ", thr_pretrain)

    print(f"ST8000_ST12000: FDR & FAR for threshold of {thr_pretrain}: ", st8000_st12000[:, 1:3][np.where(st8000_st12000[:, -1] == thr_pretrain)][0])
    fdr = st8000_st12000[:, 1][np.where(st8000_st12000[:, 2] > 0.01)][0]
    thr = st8000_st12000[:, -1][np.where(st8000_st12000[:, 2] > 0.01)][0]
    print("ST8000_ST12000: FDR for FAR = 0.01: ", fdr, " with threshold: ", thr)

    print(f"ST8000_ST12000_2M: FDR & FAR for threshold of {thr_pretrain}: ", st8000_st12000_2m[:, 1:3][np.where(st8000_st12000_2m[:, -1] == thr_pretrain)][0])
    fdr = st8000_st12000_2m[:, 1][np.where(st8000_st12000_2m[:, 2] > 0.01)][0]
    thr = st8000_st12000_2m[:, -1][np.where(st8000_st12000_2m[:, 2] > 0.01)][0]
    print("ST8000_ST12000_2M: FDR for FAR = 0.01: ", fdr, " with threshold: ", thr)

    fig, ax = plt.subplots(1, 1)

    ax.plot(st8000[:, 2], st8000[:, 1], label='st8000 pre-train')
    ax.plot(st8000_st12000[:, 2], st8000_st12000[:, 1], label='st8000 TFL st12000 no train')
    ax.plot(st8000_st12000_2m[:, 2], st8000_st12000_2m[:, 1], label='st8000 TFL st12000 2months train')
    # Vertical line at 0.01
    ax.axvline(x=0.01, color='r', linestyle='--')
    ax.legend()
    # Set x label
    ax.set_xlabel('FAR')
    # Set y label
    ax.set_ylabel('FDR')
    # Set title
    ax.set_title('FDR-FAR Tradeoff for pre-trained and transferred model')
    # x lim
    ax.set_xlim([0, 0.1])
    plt.show()

    name = 'st8000 pre-train'
    plot_histogram(test_score, test_label, name)

    1/0
    test_scores_path = 'results/model_test_st8000_st12000/results/test_score.pkl'
    test_labels_path = 'processed_data_st12000/full_test_label.pkl'
    test_data_path = 'processed_data_st12000/full_test_data.pkl'
    name = 'st8000 TFL st12000 no train'
    plot_histogram(test_scores_path, test_labels_path, test_data_path, name)

    test_scores_path = 'results/model_test_st8000_st12000_small/results/test_score.pkl'
    test_labels_path = 'processed_data_st12000/full_test_label.pkl'
    test_data_path = 'processed_data_st12000/full_test_data.pkl'
    name = 'st8000 TFL st12000 2months train'
    plot_histogram(test_scores_path, test_labels_path, test_data_path, name)
