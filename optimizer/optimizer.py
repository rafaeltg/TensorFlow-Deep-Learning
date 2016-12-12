import cma
import numpy as np
from pydl.models.base.supervised_model import SupervisedModel


class Optimizer(object):

    def __init__(self, cv, fit_fn):
        assert cv is not None
        assert fit_fn is not None

        self.cv = cv
        self.fit_fn = fit_fn

    def run(self, model, params_dict, x, y=None):
        raise NotImplementedError('This method should be overridden in child class')

    @staticmethod
    def _fit_supervised(x, model, params_dict, data_x, data_y, cv, fit_fn):
        model.set_params(*params_dict.get(x))

        fits = []

        for train_idxs, test_idxs in cv.split(data_x, data_y):
            x_train, y_train = data_x[train_idxs], data_y[train_idxs]
            x_test, y_test = data_x[test_idxs], data_y[test_idxs]

            model.fit(x_train=x_train, y_train=y_train)
            y_pred = model.predict(x_test)

            fits.append(fit_fn(y_test, y_pred))

        return np.mean(fits)

    @staticmethod
    def _fit_unsupervised(x, model, params_dict, data_x, cv, fit_fn):
        model.set_params(*params_dict.get(x))

        fits = []

        for train_idxs, test_idxs in cv.split(data_x):
            x_train, x_test = data_x[train_idxs], data_x[test_idxs]

            model.fit(x_train=x_train)
            x_rec = model.reconstruct(model.transform(x_test))

            fits.append(fit_fn(x_test, x_rec))

        return np.mean(fits)


class CMAESOptimizer(Optimizer):

    def __init__(self, cv, fit_fn, pop_size=10, sigma0=0.5, max_iter=50, verbose=-9):
        print('INIT')

        super().__init__(cv, fit_fn)

        assert pop_size > 0, 'pop_size must be greater than zero'
        assert max_iter > 0, 'max_iter must be greater than zero'
        assert sigma0 > 0 if isinstance(sigma0, float) else True, 'sigma0 must be greater than zero'

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.sigma0 = sigma0
        self.verbose = verbose

    def run(self, model, params_dict, x, y=None):

        # TODO: 'AdaptSigma' option
        es = cma.CMAEvolutionStrategy(x0=[0],
                                      sigma0=self.sigma0,
                                      inopts={
                                          'bounds': [0, 1],
                                          'popsize': self.pop_size,
                                          'verbose': self.verbose
                                      })

        if isinstance(model, SupervisedModel):
            es.optimize(objective_fct=self._fit_supervised,
                        iterations=self.max_iter,
                        args={'model': model,
                              'params_dict': params_dict,
                              'data_x': x,
                              'data_y': y,
                              'cv': self.cv,
                              'fit_fn': self.fit_fn})
        else:
            es.optimize(objective_fct=self._fit_unsupervised,
                        iterations=self.max_iter,
                        args={'model': model,
                              'params_dict': params_dict,
                              'data_x': x,
                              'cv': self.cv,
                              'fit_fn': self.fit_fn})

        return es.result()
