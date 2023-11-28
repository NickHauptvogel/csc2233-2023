# -*- coding: utf-8 -*-
import os
import pickle
import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def save_z(z, filename='z'):
    """
    save the sampled z in a txt file
    """
    for i in range(0, z.shape[1], 20):
        with open(filename + '_' + str(i) + '.txt', 'w') as file:
            for j in range(0, z.shape[0]):
                for k in range(0, z.shape[2]):
                    file.write('%f ' % (z[j][i][k]))
                file.write('\n')
    i = z.shape[1] - 1
    with open(filename + '_' + str(i) + '.txt', 'w') as file:
        for j in range(0, z.shape[0]):
            for k in range(0, z.shape[2]):
                file.write('%f ' % (z[j][i][k]))
            file.write('\n')


def get_data_dim(dataset):
    if dataset == 'Backblaze':
        return 15
    elif dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif str(dataset).startswith('machine'):
        return 38
    else:
        raise ValueError('unknown dataset '+str(dataset))


def get_data(dataset, dataset_folder, window_length, max_train_size=None, max_test_size=None, print_log=True, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print('load data of:', dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    # x_dim here with serial number as the first dimension (cut off later)
    x_dim = get_data_dim(dataset) + 1
    all_files = os.listdir(dataset_folder)
    # Shuffle the files
    random.shuffle(all_files)
    train_files = [f for f in all_files if f.endswith('_train.pkl')]
    test_files = [f for f in all_files if f.endswith('_test.pkl')]

    train_list = []
    for f in train_files:
        print(f)
        f = open(os.path.join(dataset_folder, f), "rb")
        single_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
        train_list.append(single_data)
        f.close()
    train_data = np.concatenate(train_list, axis=0)

    test_list = []
    test_label_list = []
    for file in test_files:
        print(file)
        f = open(os.path.join(dataset_folder, file), "rb")
        f_label = open(os.path.join(dataset_folder, file.replace('_test.pkl', '_test_label.pkl')), "rb")
        single_test = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        single_label = pickle.load(f_label).reshape((-1))[test_start:test_end]

        # Cut off first window_length data for each test sequence (not tested)
        single_label = single_label[window_length - 1:]
        test_list.append(single_test)
        test_label_list.append(single_label)
        f.close()
        f_label.close()

    test_data = np.concatenate(test_list, axis=0)
    test_label = np.concatenate(test_label_list, axis=0)

    if do_preprocess:
        train_data, scaler = preprocess(train_data)
        test_data, _ = preprocess(test_data, scaler)
    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", test_label.shape)
    return (train_data, None), (test_data, test_label)


def preprocess(df, scaler=None):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num()

    # normalize data
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df)
    df = scaler.transform(df)
    print('Data normalized')

    return df, scaler


def minibatch_slices_iterator(length, batch_size,
                              ignore_incomplete_batch=False):
    """
    Iterate through all the mini-batch slices.

    Args:
        length (int): Total length of data in an epoch.
        batch_size (int): Size of each mini-batch.
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of items.
            (default :obj:`False`)

    Yields
        slice: Slices of each mini-batch.  The last mini-batch may contain
               less indices than `batch_size`.
    """
    start = 0
    stop1 = (length // batch_size) * batch_size
    while start < stop1:
        yield slice(start, start + batch_size, 1)
        start += batch_size
    if not ignore_incomplete_batch and start < length:
        yield slice(start, length, 1)


class BatchSlidingWindow(object):
    """
    Class for obtaining mini-batch iterators of sliding windows.

    Each mini-batch will have `batch_size` windows.  If the final batch
    contains less than `batch_size` windows, it will be discarded if
    `ignore_incomplete_batch` is :obj:`True`.

    Args:
        array_size (int): Size of the arrays to be iterated.
        window_size (int): The size of the windows.
        batch_size (int): Size of each mini-batch.
        excludes (np.ndarray): 1-D `bool` array, indicators of whether
            or not to totally exclude a point.  If a point is excluded,
            any window which contains that point is excluded.
            (default :obj:`None`, no point is totally excluded)
        shuffle (bool): If :obj:`True`, the windows will be iterated in
            shuffled order. (default :obj:`False`)
        ignore_incomplete_batch (bool): If :obj:`True`, discard the final
            batch if it contains less than `batch_size` number of windows.
            (default :obj:`False`)
    """

    def __init__(self, array_size, window_size, batch_size, excludes=None,
                 shuffle=False, ignore_incomplete_batch=False):
        # check the parameters
        if window_size < 1:
            raise ValueError('`window_size` must be at least 1')
        if array_size < window_size:
            raise ValueError('`array_size` must be at least as large as '
                             '`window_size`')
        if excludes is not None:
            excludes = np.asarray(excludes, dtype=np.bool)
            expected_shape = (array_size,)
            if excludes.shape != expected_shape:
                raise ValueError('The shape of `excludes` is expected to be '
                                 '{}, but got {}'.
                                 format(expected_shape, excludes.shape))

        # compute which points are not excluded
        if excludes is not None:
            mask = np.logical_not(excludes)
        else:
            mask = np.ones([array_size], dtype=np.bool)
        mask[: window_size - 1] = False
        where_excludes = np.where(excludes)[0]
        for k in range(1, window_size):
            also_excludes = where_excludes + k
            also_excludes = also_excludes[also_excludes < array_size]
            mask[also_excludes] = False

        # generate the indices of window endings
        indices = np.arange(array_size)[mask]
        self._indices = indices.reshape([-1, 1])

        # the offset array to generate the windows
        self._offsets = np.arange(-window_size + 1, 1)

        # memorize arguments
        self._array_size = array_size
        self._window_size = window_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._ignore_incomplete_batch = ignore_incomplete_batch

    def get_iterator(self, arrays):
        """
        Iterate through the sliding windows of each array in `arrays`.

        This method is not re-entrant, i.e., calling :meth:`get_iterator`
        would invalidate any previous obtained iterator.

        Args:
            arrays (Iterable[np.ndarray]): 1-D arrays to be iterated.

        Yields:
            tuple[np.ndarray]: The windows of arrays of each mini-batch.
        """
        # check the parameters
        arrays = tuple(np.asarray(a) for a in arrays)
        if not arrays:
            raise ValueError('`arrays` must not be empty')

        # shuffle if required
        if self._shuffle:
            np.random.shuffle(self._indices)

        # iterate through the mini-batches
        for s in minibatch_slices_iterator(
                length=len(self._indices),
                batch_size=self._batch_size,
                ignore_incomplete_batch=self._ignore_incomplete_batch):
            idx = self._indices[s] + self._offsets
            batches = tuple(a[idx] if len(a.shape) == 1 else a[idx, :] for a in arrays)
            new_batches = []
            # Exclude samples that are across model IDs
            for batch in batches:
                clean_batch = []
                for i in range(len(batch)):
                    if np.any(batch[i, :, 0] != batch[i, 0, 0]):
                        continue
                    clean_batch.append(batch[i, :, 1:])
                batch = np.asarray(clean_batch)
                # print('batch_x:', batch_x.shape)
                if len(batch) == 0:
                    continue
                new_batches.append(batch)
            yield tuple(new_batches)
