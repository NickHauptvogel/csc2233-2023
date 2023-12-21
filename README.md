# HDD Failure Prediction using Transfer Learning

## CSC2233 Final Project 2023
Nick Hauptvogel (*hauptvog*), Carolina Villamizar (*villam26*)



This repository is based on *OmniAnomaly*, a stochastic recurrent neural network model which combines Gated Recurrent Units (GRU) or Long-Short-Term Memory (LSTM) Cells with a Variational auto-encoder (VAE). Its core idea is to learn the normal patterns of multivariate time series and uses the reconstruction probability to do anomaly judgment.

The original repository can be found here: https://github.com/NetManAIOps/OmniAnomaly/tree/master


## Getting Started

#### Clone the repo

```
git clone https://github.com/NickHauptvogel/csc2233-2023.git && cd csc2233-2023
```

#### Get data

You can get the public datasets (Backblaze) at: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data

#### Install dependencies (with python 3.5, 3.6) 

(virtualenv is recommended)

```shell
# CPU
pip install -r requirements-cpu.txt
# GPU (CUDA)
pip install -r requirements.txt
```

#### Preprocess the data

```shell
# Convert the data to csv per drive
python TODO
# Convert data into a single pickle file with train/test split
python data_preprocess.py --dataset_folder <dataset_folder> --output_folder <output_folder>
```

where `<dataset_folder>` is the folder of the dataset, and `<output_folder>` is the folder to save the preprocessed data.

```shell
# Use PCA to remove drives with high deviation 
python pca.py --dataset_folder <dataset_folder>
```

where `<dataset_folder>` is the folder of the dataset (same as output_folder in data_preprocess.py).

#### Run the code

Use the provided jupyter notebook `ModelTraining.ipynb` to run the code (either locally, then skip the checkout in the notebook). 
All parameters are set in the notebook.
