from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import validator.cv_methods as valid
from pydl.models.base.supervised_model import SupervisedModel
from validator.cv_metrics import available_metrics


class ModelValidator(object):

    """"""

    def __init__(self, method=None, **kwargs):

        """
        :param method:
        :param kwargs:
        """

        assert method is not None, 'Missing method name!'
        self.cv = valid.get_cv_method(method, **kwargs)

    def run(self, model, x=None, y=None, metrics=list([]), verbose=True):

        """
        :param model:
        :param x:
        :param y:
        :param metrics:
        :param verbose:
        :return:
        """

        assert model is not None, 'Missing model'
        assert x is not None, 'Missing dataset x!'

        if isinstance(model, SupervisedModel):
            cv_metrics = self._run_supervised_cv(model, x, y, metrics, verbose)
        else:
            cv_metrics = self._run_unsupervised_cv(model, x, metrics, verbose)

        # Calculate mean and sd for each metric
        results = {}
        for k, v in cv_metrics.items():
            results[k] = {
                'values': v,
                'mean': np.mean(v),
                'sd': np.std(v)
            }

        return results

    def _run_supervised_cv(self, model, x, y, metrics, verbose):
        assert y is not None, 'Missing dataset y!'

        cv_metrics = self._init_cv_metrics(metrics)

        i = 0
        for train_idxs, test_idxs in self.cv.split(x, y):
            if verbose:
                print('\nCV - %d' % i)

            x_train, y_train = x[train_idxs], y[train_idxs]
            x_test, y_test = x[test_idxs], y[test_idxs]

            model.fit(x_train=x_train, y_train=y_train)

            cv_metrics['scores'].append(model.score(x_test, y_test))

            if len(metrics) > 0:
                preds = model.predict(x_test)

                for m in metrics:
                    v = available_metrics[m](y_test, preds)
                    cv_metrics[m].append(v)

            i += 1

        return cv_metrics

    def _run_unsupervised_cv(self, model, x, metrics, verbose):

        cv_metrics = self._init_cv_metrics(metrics)

        i = 0
        for train_idxs, test_idxs in self.cv.split(x):
            if verbose:
                print('\nCV - %d' % i)

            x_train, x_test = x[train_idxs], x[test_idxs]

            model.fit(x_train=x_train)

            cv_metrics['scores'].append(model.score(x_test))

            if len(metrics) > 0:
                x_rec = model.reconstruct(model.transform(x_test))

                for m in metrics:
                    v = available_metrics[m](x_test, x_rec)
                    cv_metrics[m].append(v)

            i += 1

        return cv_metrics

    @staticmethod
    def _init_cv_metrics(metrics):
        cv_metrics = {'scores': []}
        if len(metrics) > 0:
            for m in metrics:
                assert m in available_metrics.keys(), 'Invalid metric - %s' % m
                cv_metrics[m] = list([])

        return cv_metrics

