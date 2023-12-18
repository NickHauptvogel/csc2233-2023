import json

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

from omni_anomaly.eval_methods import bf_search, bf_search_binary


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
    plt.tight_layout()
    plt.savefig(name + '_histogram.pdf')
    plt.show()


def get_bf_search_score(result_dir, test_data_path, name, thr=None, days=14, min=-2000, plot_hist=False):
    # If bf_search.pkl does not exist, generate it
    if not os.path.exists(os.path.join(result_dir, 'bf_search.pkl')) or plot_hist:
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
        def change_label(x):
            if x.iloc[-1, -1] == 1:
                x.iloc[:, -1] = 0
                x.iloc[-days:, -1] = 1
            return x
        test_df = test_df.groupby(test_df.columns[0], sort=False).apply(change_label)
        # Concatenate test_score
        test_df = pd.concat([test_df, pd.DataFrame(test_score)], axis=1)
        # Use last row as label
        test_label = test_df.iloc[:, -2].values
        # Generate bf_search.pkl
        if not os.path.exists(os.path.join(result_dir, 'bf_search.pkl')):
            arr = bf_search(test_score, test_label, min)
            pickle.dump(arr, open(os.path.join(result_dir, 'bf_search.pkl'), 'wb'))

        plot_histogram(test_score, test_label, name)

    bf_search_score = pickle.load(open(os.path.join(result_dir, 'bf_search.pkl'), 'rb'))

    # Get FDR for FAR = 0.01
    fdr = bf_search_score[:, 1][np.where(bf_search_score[:, 2] > 0.01)][0]
    thr_001 = bf_search_score[:, -1][np.where(bf_search_score[:, 2] > 0.01)][0]
    far = bf_search_score[:, 2][np.where(bf_search_score[:, -1] > thr_001)][0]
    print(name + f":\tFAR = {round(far, 4)} \t FDR = ", round(fdr, 4), "\t with threshold: ", thr_001)
    if thr is not None:
        fdr = bf_search_score[:, 1][np.where(bf_search_score[:, -1] > thr)][0]
        far = bf_search_score[:, 2][np.where(bf_search_score[:, -1] > thr)][0]
        print(name + f":\tFAR = {round(far, 4)} \t FDR = ", round(fdr, 4), "\t with threshold: ", thr)

    return bf_search_score, thr_001


def search_anomaly_days_threshold(result_dir, test_data_path, days=14, min=-2000):

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
        def change_label(x, d):
            if x.iloc[-1, -1] == 1:
                x.iloc[:, -1] = 0
                x.iloc[-d:, -1] = 1
            return x
        for i in range(1, days):
            print("Days: ", i)
            test_df_labeled = test_df.copy()
            test_df_labeled = test_df_labeled.groupby(test_df_labeled.columns[0], sort=False).apply(change_label, d=i)
            # Concatenate test_score
            test_df_labeled = pd.concat([test_df_labeled, pd.DataFrame(test_score)], axis=1)
            # Use last row as label
            test_label = test_df_labeled.iloc[:, -2].values
            # Generate bf_search.pkl
            bf_search_binary(test_score, test_label, min)


if __name__ == '__main__':

    # Define anomaly days
    anomaly_days = 7

    # Get far-fdr for each experiment
    st8000, thr0 = get_bf_search_score('results/tuned_model_st8000_2/results', 'processed_st8000/test.pkl', 'Pre-Train', days=anomaly_days)
    st8000_exp1, thr1 = get_bf_search_score('results/tuned_model_st8000_2_exp1/results', 'processed_st12000/test.pkl', 'Experiment 1', thr0, days=anomaly_days)
    st8000_exp2a, thr2a = get_bf_search_score('results/tuned_model_st8000_2_exp2a/results', 'processed_st12000/test.pkl', 'Experiment 2 - 30 days', thr0, days=anomaly_days)
    st8000_exp2b, thr2b = get_bf_search_score('results/tuned_model_st8000_2_exp2b/results', 'processed_st12000/test.pkl', 'Experiment 2 - 10%', thr0, days=anomaly_days)
    st8000_exp2b_1p, thr2b_1p = get_bf_search_score('results/tuned_model_st8000_2_exp2b_1p/results', 'processed_st12000/test.pkl', 'Experiment 2 - 1%', thr0, days=anomaly_days, plot_hist=True)
    st8000_exp2b_0_1p, thr2b_0_1p = get_bf_search_score('results/tuned_model_st8000_2_exp2b_0_1p/results', 'processed_st12000/test.pkl', 'Experiment 2 - 0.1%', thr0, days=anomaly_days)
    st8000_exp3, thr3 = get_bf_search_score('results/tuned_model_st8000_2_exp3/results', 'processed_st12000/test.pkl', 'Experiment 3', thr0, days=anomaly_days)
    st8000_exp4, thr4 = get_bf_search_score('results/tuned_model_st8000_2_exp4/results', 'processed_st12000/test.pkl', 'Experiment 4 - 10%', thr0, days=anomaly_days)
    st8000_exp4_1p, thr4_1p = get_bf_search_score('results/tuned_model_st8000_2_exp4_1p/results', 'processed_st12000/test.pkl', 'Experiment 4 - 1%', thr0, days=anomaly_days)
    st8000_exp4_0_1p, thr4_0_1p = get_bf_search_score('results/tuned_model_st8000_2_exp4_0_1p/results', 'processed_st12000/test.pkl', 'Experiment 4 - 0.1%', thr0, days=anomaly_days)

    # Plot 1: FAR-FDR for experiment 1, 2, 3
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot(st8000[:, 2], st8000[:, 1], label='Pre-Train', color='blue')
    ax.plot(st8000_exp1[:, 2], st8000_exp1[:, 1], label='Exp. 1 - No Train', color='red')
    ax.plot(st8000_exp2a[:, 2], st8000_exp2a[:, 1], label='Exp. 2 - 30d/disk', color='purple')
    ax.plot(st8000_exp2b[:, 2], st8000_exp2b[:, 1], label='Exp. 2 - 10% data', color='green')
    ax.plot(st8000_exp3[:, 2], st8000_exp3[:, 1], label='Exp. 3 - 100% data', color='orange')
    # Vertical line at 0.01
    ax.axvline(x=0.01, color='grey', linestyle='--')
    ax.legend(loc='lower right')
    ax.set_xlabel('FAR')
    ax.set_ylabel('FDR')
    ax.set_title('FDR-FAR Tradeoff')
    ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig('exp1_exp2a_exp2b_exp3.pdf')
    plt.show()

    # Plot 2: FAR-FDR for experiment 2, 4
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot(st8000_exp2b[:, 2], st8000_exp2b[:, 1], label='Exp. 2 - 10% data', color='green')
    ax.plot(st8000_exp2b_1p[:, 2], st8000_exp2b_1p[:, 1], label='Exp. 2 - 1% data', color='red')
    ax.plot(st8000_exp2b_0_1p[:, 2], st8000_exp2b_0_1p[:, 1], label='Exp. 2 - 0.1% data', color='orange')
    ax.plot(st8000_exp4[:, 2], st8000_exp4[:, 1], label='Exp. 4 - 10% data', color='green', linestyle='--')
    ax.plot(st8000_exp4_1p[:, 2], st8000_exp4_1p[:, 1], label='Exp. 4 - 1% data', color='red', linestyle='--')
    # Vertical line at 0.01
    ax.axvline(x=0.01, color='grey', linestyle='--')
    ax.legend(loc='lower right')
    ax.set_xlabel('FAR')
    ax.set_ylabel('FDR')
    ax.set_title('FDR-FAR Tradeoff')
    ax.set_xlim([0, 0.1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig('exp2_exp4.pdf')
    plt.show()
