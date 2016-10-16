from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import validator.validation_method as valid
from pydl.models.base.supervised_model import SupervisedModel
from validator.validation_metrics import available_metrics


class ModelValidator(object):

    """"""

    def __init__(self, method='kfold', **kwargs):

        """
        :param method:
        :param kwargs:
        """

        assert method in ['kfold', 'skfold', 'shuffle_split', 'split', 'time_series']

        if method == 'kfold':
            self.method = valid.KFoldValidation(**kwargs)
        elif method == 'skfold':
            self.method = valid.StratifiedKFoldValidation(**kwargs)
        elif method == 'shuffle_split':
            self.method = valid.ShuffleSplitValidation(**kwargs)
        elif method == 'split':
            self.method = valid.SplitValidation(**kwargs)
        else:
            self.method = valid.TimeSeriesValidation(**kwargs)

    def run(self, model, x=None, y=None, metrics=list([])):

        """
        :param model:
        :param x:
        :param y:
        :param metrics:
        :return:
        """

        assert isinstance(model, SupervisedModel), 'Invalid model.'
        assert all([m in available_metrics.keys() for m in metrics])
        assert x is not None, 'Missing dataset x.'
        assert y is not None, 'Missing dataset y.'

        scores = []
        cv_metrics = None

        if len(metrics) > 0:
            cv_metrics = {}
            for m in metrics:
                cv_metrics[m] = []

        i = 0
        for train_idxs, test_idxs in self.method.get_cv_folds(x, y):
            print('\nCV - %d' % i)
            print(len(test_idxs))

            x_train, y_train = x[train_idxs], y[train_idxs]
            x_test, y_test = x[test_idxs], y[test_idxs]

            model.fit(x_train=x_train, y_train=y_train)

            s = model.score(x_test, y_test)
            scores.append(s)
            print('> Test score = %f' % s)

            if cv_metrics:
                preds = model.predict(x_test)

                for m in metrics:
                    v = available_metrics[m](y_test, preds)
                    print('> %s = %f' % (m, v))
                    cv_metrics[m].append(v)

            i += 1

        return {
            'scores': scores,
            'metrics': cv_metrics
        }
