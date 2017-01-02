import cma
import numpy as np
import multiprocessing as mp
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
    def fit_supervised(x, model, params_dict, data_x, data_y, cv, fit_fn):
        model.set_params(**params_dict.get(x))
        fits = []

        for train_idxs, test_idxs in cv.split(data_x, data_y):
            x_train, y_train = data_x[train_idxs], data_y[train_idxs]
            x_test, y_test = data_x[test_idxs], data_y[test_idxs]

            model.fit(x_train=x_train, y_train=y_train)
            y_pred = model.predict(x_test)

            fits.append(fit_fn(y_test, y_pred))

        return np.mean(fits)

    @staticmethod
    def fit_unsupervised(x, model, params_dict, data_x, cv, fit_fn):
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
        super().__init__(cv, fit_fn)

        assert pop_size > 0, 'pop_size must be greater than zero'
        assert max_iter > 0, 'max_iter must be greater than zero'
        assert sigma0 > 0 if isinstance(sigma0, float) else True, 'sigma0 must be greater than zero'

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.sigma0 = sigma0
        self.verbose = verbose

    def run(self, model, params_dict, x, y=None, max_thread=4):

        # TODO: 'AdaptSigma' option
        es = cma.CMAEvolutionStrategy(x0=[0] * params_dict.get_size(),
                                      sigma0=self.sigma0,
                                      inopts={
                                          'maxiter': self.max_iter,
                                          'bounds': [0, 1],
                                          'popsize': self.pop_size,
                                          'verbose': self.verbose
                                      })

        if isinstance(model, SupervisedModel):
            obj_f = self.fit_supervised
            args = (model,
                    params_dict,
                    x,
                    y,
                    self.cv,
                    self.fit_fn)
        else:
            obj_f = self.fit_unsupervised
            args = (model,
                    params_dict,
                    x,
                    self.cv,
                    self.fit_fn)

        if max_thread == 1:
            es.optimize(objective_fct=obj_f,
                        args=args)
        else:
            with mp.Pool(max_thread) as pool:
                while not es.stop():
                    X = es.ask()
                    f_values = pool.starmap(func=obj_f,
                                            iterable=[(_x, *args) for _x in X])
                                            #chunksize=es.popsize/max_thread)
                    # use chunksize parameter as es.popsize/len(pool)?
                    es.tell(X, f_values)
                    es.disp()
                    es.logger.add()

        return es.result()
