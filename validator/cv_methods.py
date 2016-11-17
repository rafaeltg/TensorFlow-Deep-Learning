from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import floor

from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold, ShuffleSplit, TimeSeriesSplit


class TrainTestSplitCV:

    def __init__(self, test_size=0.3):
        self.test_size = test_size

    def split(self, X, y=None):
        if X is None:
            raise AttributeError('X cannot be empty!')

        n = len(X)
        train_size = floor(n * (1 - self.test_size))
        yield slice(0, train_size, 1), slice(train_size, n, 1)


class TimeSeriesCV:

    def __init__(self, n_splits=1, fixed=False):
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        self.fixed = fixed

    def split(self, X, y=None):
        if not self.fixed:
            return self.cv.split(X=X, y=y)

        prev_fold = []
        for train, test in self.cv.split(X, y):
            train_idxs = set(train).difference(set(prev_fold))
            prev_fold = train
            yield train_idxs, test


def get_cv_method(method, **kwargs):

    if method == 'kfold':
        return KFold(**kwargs)
    elif method == 'skfold':
        return StratifiedKFold(**kwargs)
    elif method == 'loo':
        return LeaveOneOut()
    elif method == 'shuffle_split':
        return ShuffleSplit(**kwargs)
    elif method == 'split':
        return TrainTestSplitCV(**kwargs)
    elif method == 'time_series':
        return TimeSeriesCV(**kwargs)
    else:
        raise AttributeError('Invalid method - %s!' % method)
