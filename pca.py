# Import PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_folder', type=str, default='processed_st12000', help='dataset used for training')
args = parser.parse_args()

if __name__ == '__main__':

    train_data_path = args.dataset_folder
    train_data = pickle.load(open(os.path.join(train_data_path, 'train.pkl'), 'rb'))

    train_data_bare = train_data[:, 1:]
    pca = PCA()
    scaler = StandardScaler()
    train_data_bare = scaler.fit_transform(train_data_bare)
    pca.fit(train_data_bare)

    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    # Plot the cumulative sum of explained variances
    plt.step(features, np.cumsum(pca.explained_variance_ratio_), where='mid', label='Cumulative variance')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()
    # Get number of components explaining 50% of variance
    n_components = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.5)[0][0] + 1
    print('Number of components explaining 50% of variance: ' + str(n_components))

    scores = train_data_bare.dot(pca.components_[0:n_components, :].T)
    # Square each
    scores = scores**2
    # Divide by eigenvalues
    scores = scores / pca.explained_variance_[0:n_components]
    # Sum over all components
    scores = np.sum(scores, axis=1)

    # Plot histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(scores, bins=np.arange(0, 20, 0.5), density=True, alpha=0.5, color='b')
    ax.set_xlabel('Anomaly score')
    ax.set_ylabel('Density')

    plt.show()

    # Concatenate train_data and scores
    drives_score = np.concatenate((np.expand_dims(train_data[:, 0], axis=1), np.expand_dims(scores, axis=1)), axis=1)
    # Get maximum score per drive
    drives_score = pd.DataFrame(drives_score)
    # Group by drive and get maximum score
    drives_score = drives_score.groupby(0).max()
    # Get 10% of drives with highest score
    drives_score = drives_score.sort_values(by=1, ascending=False)
    drives_score = drives_score.iloc[0:int(len(drives_score) * 0.1), :]
    # Get serial numbers
    drives_score = drives_score.reset_index()
    drives = drives_score.iloc[:, 0]

    # Filter train_data by drives
    train_data = pd.DataFrame(train_data)
    train_data = train_data[~train_data.iloc[:, 0].isin(drives)]
    train_data = train_data.values

    # Save train_data
    pickle.dump(train_data, open(os.path.join(train_data_path, 'train_pca.pkl'), 'wb'))




