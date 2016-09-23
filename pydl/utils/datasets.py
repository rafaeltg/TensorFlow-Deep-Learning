import collections
import csv
import os

import numpy as np
from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'test', 'validation'])


def load_csv(filename, data_type=np.float64, has_header=True):
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)

        if has_header:
            _ = next(data_file)

        data = []
        for ir in data_file:
            data.append(np.asarray(ir, dtype=data_type))

        data = np.asarray(data, dtype=data_type)

    return data


def load_npy(filename):
    return np.load(filename) if filename != '' else None


def load_dataset(x_path, y_path='', y_dtype=np.float64, has_header=True):

    if x_path == '':
        return Dataset(data=None, target=None)

    if os.path.splitext(x_path)[1] == '.csv':
        x = load_csv(filename=x_path, has_header=has_header)

    else:
        x = load_npy(x_path)

    if y_path is '':
        y = None

    elif os.path.splitext(y_path)[1] == '.csv':
        y = load_csv(filename=y_path, data_type=y_dtype, has_header=has_header)

    else:
        y = load_npy(y_path)

    return Dataset(data=x, target=y)


def load_datasets(train_dataset,
                  train_labels='',
                  test_dataset='',
                  test_labels='',
                  valid_dataset='',
                  valid_labels='',
                  labels_dtype=np.float64,
                  has_header=True):

    """
    """

    assert train_dataset != ''

    return Datasets(train=load_dataset(train_dataset, train_labels, labels_dtype, has_header),
                    test=load_dataset(test_dataset, test_labels, labels_dtype, has_header),
                    validation=load_dataset(valid_dataset, valid_labels, labels_dtype, has_header))