from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import floor

from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, TimeSeriesSplit


class ValidateMethod(object):

    def __init__(self, **kwargs):
        self.cv = None
        self._get_cv(**kwargs)

    def _get_cv(self, **kwargs):
        pass

    def get_cv_folds(self, x, y):
        return self.cv.split(X=x, y=y)


class KFoldValidation(ValidateMethod):

    def _get_cv(self, **kwargs):
        self.cv = KFold(n_splits=kwargs.get('n_split'),
                        shuffle=kwargs.get('shufle'),
                        random_state=kwargs.get('random_state'))


class StratifiedKFoldValidation(ValidateMethod):

    def _get_cv(self, **kwargs):
        self.cv = StratifiedKFold(n_splits=kwargs.get('n_split'),
                                  shuffle=kwargs.get('shufle'),
                                  random_state=kwargs.get('random_state'))


class ShuffleSplitValidation(ValidateMethod):

    def _get_cv(self, **kwargs):
        self.cv = ShuffleSplit(n_splits=kwargs.get('n_split'),
                               test_size=kwargs.get('test_size'),
                               random_state=kwargs.get('random_state'))


class SplitValidation(ValidateMethod):

    def _get_cv(self, **kwargs):
        self.test_size = kwargs.get('test_size', None)
        assert self.test_size is not None, 'Missing test_size.'

    def get_cv_folds(self, x, y):
        n = len(y)
        train_size = floor(n * (1 - self.test_size))
        yield slice(0, train_size, 1), slice(train_size, n, 1)


class TimeSeriesValidation(ValidateMethod):

    def _get_cv(self, **kwargs):
        self.cv = TimeSeriesSplit(n_splits=kwargs.get('n_splits'))
