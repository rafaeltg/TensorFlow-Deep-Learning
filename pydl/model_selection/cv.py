import gc
import numpy as np
import multiprocessing as mp
from .methods import get_cv_method
from .scorer import get_scorer
from ..models.utils import model_from_config
from ..utils import dump_np_data_set, free_mmap_data_set


class CV(object):

    """
        Cross-Validation
    """

    def __init__(self, method, **kwargs):
        self.cv = get_cv_method(method, **kwargs)

    def run(self, model, x, y=None, scoring=None, max_threads=1, in_memory=False):

        # get scorers
        if scoring is not None:
            if isinstance(scoring, list):
                scorers_fn = dict([(self.get_scorer_name(k), get_scorer(k)) for k in scoring])
            else:
                scorers_fn = dict([(self.get_scorer_name(scoring), get_scorer(scoring))])
        else:
            # By default uses the model loss function as scoring function
            scorers_fn = dict([(model.get_loss_func(), get_scorer(model.get_loss_func()))])

        try:
            model_cfg = model.to_json()

            if in_memory:
                x, y = dump_np_data_set(x, y)
                gc.collect()

            if y is None:
                args = [(model_cfg, train, test, x, scorers_fn) for train, test in self.cv.split(x, y)]
                cv_fn = self._do_unsupervised_cv
            else:
                args = [(model_cfg, train, test, x, y, scorers_fn) for train, test in self.cv.split(x, y)]
                cv_fn = self._do_supervised_cv

            if max_threads == 1:
                cv_results = [cv_fn(*a) for a in args]
            else:
                max_threads = min(max_threads, len(args))
                with mp.Pool(max_threads) as pool:
                    cv_results = pool.starmap(func=cv_fn, iterable=args, chunksize=len(args)//max_threads)

        finally:
            if in_memory:
                free_mmap_data_set(x, y)

        return self._consolidate_cv_scores(cv_results)

    @staticmethod
    def _do_supervised_cv(model_cfg, train, test, x, y, scorers_fn):
        model = model_from_config(model_cfg)
        model.fit(x[train], y[train])
        x_test, y_test = x[test], y[test]
        cv_result = dict([(k, scorer(model, x_test, y_test)) for k, scorer in scorers_fn.items()])
        return cv_result

    @staticmethod
    def _do_unsupervised_cv(model_cfg, train, test, x, scorers_fn):
        model = model_from_config(model_cfg)
        model.fit(x[train])
        x_test = x[test]
        cv_result = dict([(k, scorer(model, x_test)) for k, scorer in scorers_fn.items()])
        return cv_result

    @staticmethod
    def _consolidate_cv_scores(cv_results):
        cv_scores = {}
        for k in cv_results[0].keys():
            scores = [result[k] for result in cv_results]
            cv_scores[k] = {
                'values': scores,
                'mean': np.mean(scores),
                'sd': np.std(scores)
            }
        return cv_scores

    @staticmethod
    def get_scorer_name(scorer):
        if isinstance(scorer, str):
            return scorer
        elif hasattr(scorer, '__call__'):
            return scorer.__name__


