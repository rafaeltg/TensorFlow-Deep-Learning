from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import floor
import sklearn.cross_validation as cv


class ValidateMethod(object):

    def get_cv_folds(self, y):
        pass


class KFoldValidation(ValidateMethod):

    def __init__(self, k=10):
        self.k = k

    def get_cv_folds(self, y):
        return cv.KFold(len(y), self.k)


class StratifiedKFoldValidation(ValidateMethod):

    def __init__(self, k=10):
        self.k = k

    def get_cv_folds(self, y):
        return cv.StratifiedKFold(y, self.k)


class LOOValidation(ValidateMethod):

    def get_cv_folds(self, y):
        return cv.LeaveOneOut(len(y))


class ShuffleSplitValidation(ValidateMethod):

    def __init__(self, test_size=0.25, n_iter=5):
        self.test_size = test_size
        self.n_iter = n_iter

    def get_cv_folds(self, y):
        return cv.ShuffleSplit(len(y), n_iter=self.n_iter, test_size=self.test_size)


class SplitValidation(ValidateMethod):

    def __init__(self, test_size=0.3):
        self.test_size = test_size

    def get_cv_folds(self, y):
        n = len(y)
        train_size = floor(n * (1 - self.test_size))
        yield slice(0, train_size, 1), slice(train_size, n, 1)
