import multiprocessing as mp
import numpy as np

from ..models.base import SupervisedModel
from .methods import *
from .metrics import available_metrics


class CV(object):

    """
        Cross-Validation
    """

    def __init__(self, method, **kwargs):
        self.cv = get_cv_method(method, **kwargs)

    def run(self, model, x, y=None, metrics=list([]), pp=None, max_thread=1):

        """
        :param model:
        :param x:
        :param y:
        :param metrics:
        :param pp: Preprocessing object (sklearn.preprocessing)
        :param max_thread:
        :return:
        """

        metrics_fn = {}
        for m in metrics:
            assert m in available_metrics.keys(), 'Invalid metric - %s' % m
            metrics_fn[m] = available_metrics[m]

        args = []
        if isinstance(model, SupervisedModel):
            assert y is not None, 'Missing y!'

            cv_fn = self._supervised_cv
            for train, test in self.cv.split(x, y):
                args.append((model.copy(),
                             x[train],
                             y[train],
                             x[test],
                             y[test],
                             pp,
                             metrics_fn))
        else:
            cv_fn = self._unsupervised_cv
            for train, test in self.cv.split(x):
                args.append((model,
                             x[train],
                             x[test],
                             metrics_fn))

        cv_results = []
        if max_thread == 1:
            for fn_args in args:
                cv_results.append(cv_fn(*fn_args))

        else:
            with mp.Pool(max_thread) as pool:
                cv_results = pool.starmap(func=cv_fn, iterable=args)

        return self._consolidate_cv_metrics(cv_results)

    @staticmethod
    def _supervised_cv(model, x_train, y_train, x_test, y_test, pp, metrics_fn):

        # Data preprocessing
        if pp:
            x_train = pp.fit_transform(x_train)
            x_test = pp.transform(x_test)

        model.fit(x_train=x_train, y_train=y_train)

        cv_result = {model.get_loss_func(): model.score(x_test, y_test)}

        if len(metrics_fn) > 0:
            y_pred = model.predict(x_test)

            for k, fn in metrics_fn.items():
                if k == 'log_loss':
                    res = fn(y_test, model.predict(x_test, predic_probs=True))
                else:
                    res = fn(y_test, y_pred)
                cv_result[k] = res

        return cv_result

    @staticmethod
    def _unsupervised_cv(model, x_train, x_test, pp, metrics_fn):
        model.fit(x_train=x_train)

        cv_result = {model.get_loss_func(): model.score(x_test)}

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

