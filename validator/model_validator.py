from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import multiprocessing as mp

import validator.cv_methods as valid
from pydl.models.base.supervised_model import SupervisedModel
from validator.cv_metrics import available_metrics


class ModelValidator(object):

    """

    """

    def __init__(self, method=None, **kwargs):
        assert method is not None, 'Missing method name!'
        self.cv = valid.get_cv_method(method, **kwargs)

    def run(self, model, x=None, y=None, metrics=list([]), max_thread=2):

        """
        :param model:
        :param x:
        :param y:
        :param metrics:
        :param max_thread:
        :return:
        """

        assert model is not None, 'Missing model!'
        assert x is not None, 'Missing x!'

        metrics_fm = {}
        for m in metrics:
            assert m in available_metrics.keys(), 'Invalid metric - %s' % m
            metrics_fm[m] = available_metrics[m]

        args = []
        if isinstance(model, SupervisedModel):
            assert y is not None, 'Missing y!'

            cv_fn = self._supervised_cv
            for train, test in self.cv.split(x, y):
                args.append((model,
                             x[train],
                             y[train],
                             x[test],
                             y[test],
                             metrics_fm))
        else:
            cv_fn = self._unsupervised_cv
            for train, test in self.cv.split(x):
                args.append((model,
                             x[train],
                             x[test],
                             metrics_fm))

        with mp.Pool(max_thread) as pool:
            cv_results = pool.starmap(func=cv_fn,
                                      iterable=args)

        return self._consolidate_cv_metrics(cv_results)

    @staticmethod
    def _supervised_cv(model, x_train, y_train, x_test, y_test, metrics_fn):
        model.fit(x_train=x_train, y_train=y_train)

        cv_result = {'scores': model.score(x_test, y_test)}

        if len(metrics_fn) > 0:
            y_pred = model.predict(x_test)

            for k, fn in metrics_fn.items():
                cv_result[k] = fn(y_test, y_pred)

        return cv_result

    @staticmethod
    def _unsupervised_cv(model, x_train, x_test, metrics_fn):
        model.fit(x_train=x_train)

        cv_result = {'scores': model.score(x_test)}

        if len(metrics_fn) > 0:
            x_rec = model.reconstruct(model.transform(x_test))

            for k, fn in metrics_fn.items():
                cv_result[k] = fn(x_test, x_rec)

        return cv_result

    @staticmethod
    def _consolidate_cv_metrics(cv_results):
        cv_metrics = {m: [] for m in cv_results[0].keys()}
        for result in cv_results:
            for k, v in result.items():
                cv_metrics[k].append(v)

        # Calculate mean and sd for each metric
        cv_result = {}
        for k, v in cv_metrics.items():
            cv_result[k] = {
                'values': v,
                'mean': np.mean(v),
                'sd': np.std(v)
            }

        return cv_result

