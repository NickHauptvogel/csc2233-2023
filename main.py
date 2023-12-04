# -*- coding: utf-8 -*-
import json
import logging
import os
import pickle
import sys
import time
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pprint import pformat, pprint

import numpy as np
import tensorflow as tf
from tfsnippet.examples.utils import MLResults, print_with_title
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict, register_config_arguments, Config

from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.training import Trainer
from omni_anomaly.utils import get_data, save_z

import wandb

def get_sweepID():
    # Define hyperparameter search space
    sweep_configuration = {
        "method": "grid",
        "name": "CSC2233 Hyperparameter Search 1",
        "metric": {
            "name": "best_valid_loss",
            "goal": "minimize"
        },
        "parameters": {
            "rnn_cell": {
                "values": ['GRU', 'LSTM']
            },
            "rnn_num_hidden": {
                "values": [100, 500]
            },
            "window_length": {
                "values": [15, 25, 50]
            },
            "dense_dim": {
                "values": [100, 500]
            },
            "posterior_flow_type": {
                "values": ['nf', 'None']
            },
            "nf_layers": {
                "values": [5, 20]
            }
        }
    }
    sweepID = wandb.sweep(sweep_configuration, project="csc2233")

    return sweepID

class ExpConfig(Config):
    # dataset configuration
    dataset_folder = 'processed'
    x_dim = 15

    # model architecture configuration
    use_connected_z_q = False
    use_connected_z_p = False

    # model parameters
    z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 25
    dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 100
    train_start = 0
    train_portion = None  # `None` means full train set
    train_days_per_disk = None  # `None` means full train set. Each disk only gets `train_no_days` days of data
    batch_size = 512
    initial_lr = 0.0001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 20
    std_epsilon = 1e-4
    early_stopping_patience = 10

    hyperparameter_search = False
    sweepID = None

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 512

    valid_portion = 0.2
    gradient_clip_norm = 10.

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = False  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = 'model'
    restore_dir = None  # If not None, restore variables from this dir
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'
    scaler_path = None # Path to pickle file containing the scaler


def main():
    if config.hyperparameter_search:
        run = wandb.init()
        for k, v in wandb.config.items():
            config.__setattr__(k, v)

    save_dir = os.path.join('results', config.save_dir + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    result_dir = os.path.join(save_dir, 'results')
    # open the result object and prepare for result directories if specified
    results = MLResults(result_dir)
    results.save_config(config)  # save experiment settings for review
    results.make_dirs(save_dir, exist_ok=True)

    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    if config.hyperparameter_search:
        wandb.log({"config": config.to_dict()})

    logging.basicConfig(
        level='INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # prepare the data
    (x_train, _), (x_test, y_test), scaler = get_data(config.dataset_folder,
                                                      config.window_length,
                                                      train_portion=config.train_portion,
                                                      train_start=config.train_start,
                                                      scaler_path=config.scaler_path,
                                                      train_days_per_disk=config.train_days_per_disk)
    with open(os.path.join(result_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # construct the model under `variable_scope` named 'model'
    with tf.variable_scope('model') as model_vs:
        model = OmniAnomaly(config=config, name="model")

        # construct the trainer
        trainer = Trainer(model=model,
                          model_vs=model_vs,
                          max_epoch=config.max_epoch,
                          batch_size=config.batch_size,
                          valid_batch_size=config.test_batch_size,
                          initial_lr=config.initial_lr,
                          lr_anneal_epochs=config.lr_anneal_epoch_freq,
                          lr_anneal_factor=config.lr_anneal_factor,
                          early_stopping_patience=config.early_stopping_patience,
                          grad_clip_norm=config.gradient_clip_norm,
                          save_dir=save_dir)

        # construct the predictor
        predictor = Predictor(model, batch_size=config.batch_size, n_z=config.test_n_z,
                              last_point_only=True)

        with tf.Session().as_default():

            if config.restore_dir is not None:
                # Restore variables from `save_dir`.
                saver = VariableSaver(get_variables_as_dict(model_vs), config.restore_dir)
                saver.restore()
                print('Variables restored from {}.'.format(config.restore_dir))
                # Open results
                try:
                    results_json = json.load(open(os.path.join(result_dir, 'result.json'), 'r'))
                    start_val_loss = results_json['best_valid_loss']
                except:
                    start_val_loss = float('inf')
            else:
                start_val_loss = float('inf')
            print('Start val loss: {}'.format(start_val_loss))

            if config.max_epoch > 0:
                # train the model
                train_start = time.time()
                best_valid_metrics = trainer.fit(x_train, valid_portion=config.valid_portion, start_val_loss=start_val_loss, wandb_log=config.hyperparameter_search)
                train_time = (time.time() - train_start) / config.max_epoch
                best_valid_metrics.update({
                    'train_time': train_time
                })
            else:
                best_valid_metrics = {}

            train_score, train_z, train_pred_speed = predictor.get_score(x_train)
            if config.train_score_filename is not None:
                with open(os.path.join(result_dir, config.train_score_filename), 'wb') as file:
                    pickle.dump(train_score, file)
            if config.save_z:
                save_z(train_z, 'train_z')

            if x_test is not None:
                # get score of test set
                test_start = time.time()
                test_score, test_z, pred_speed = predictor.get_score(x_test)
                test_time = time.time() - test_start
                if config.save_z:
                    save_z(test_z, 'test_z')
                best_valid_metrics.update({
                    'pred_time': pred_speed,
                    'pred_total_time': test_time
                })
                if config.test_score_filename is not None:
                    with open(os.path.join(result_dir, config.test_score_filename), 'wb') as file:
                        pickle.dump(test_score, file)

            if save_dir is not None:
                # save the variables
                var_dict = get_variables_as_dict(model_vs)
                saver = VariableSaver(var_dict, save_dir)
                saver.save()
            print('=' * 30 + 'result' + '=' * 30)
            pprint(best_valid_metrics)
            if config.hyperparameter_search:
                wandb.log(best_valid_metrics)
                run.finish()


if __name__ == '__main__':

    # get config obj
    config = ExpConfig()
    # parse the arguments
    arg_parser = ArgumentParser()
    register_config_arguments(config, arg_parser)
    arg_parser.parse_args(sys.argv[1:])

    with warnings.catch_warnings():
        # suppress DeprecationWarning from NumPy caused by codes in TensorFlow-Probability
        warnings.filterwarnings("ignore", category=DeprecationWarning, module='numpy')
        # get sweep id for hyperparameter optimization
        if config.hyperparameter_search:
            if config.sweepID is not None:
                sweep_ID = config.sweepID
            else:
                sweep_ID = get_sweepID()
            print('Sweep ID: {}'.format(sweep_ID))
            wandb.agent(sweep_ID, function=main, project="csc2233")
        else:
            main()
