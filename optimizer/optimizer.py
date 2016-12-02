import cma
from pydl.models.base.supervised_model import SupervisedModel


class Optimizer(object):

    def run(self, model, params_dict, x, y=None):
        raise NotImplementedError('This method should be overridden in child class')

    @staticmethod
    def _fit_supervised(x, model, params_dict, data_x, data_y):
        # TODO
        pass

    @staticmethod
    def _fit_unsupervised(x, model, params_dict, data_x):
        # TODO
        pass


class CMAESOptimizer(Optimizer):

    def __init__(self, pop_size, sigma0, max_iter, verbose=-9):
        assert pop_size > 0, 'pop_size must be greater than zero'
        assert max_iter > 0, 'max_iter must be greater than zero'
        assert sigma0 > 0 if isinstance(sigma0, float) else True, 'sigma0 must be greater than zero'

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.sigma0 = sigma0
        self.verbose = verbose

    def run(self, model, params, x, y=None):

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
                              'params_dict': params,
                              'data_x': x,
                              'data_y': y})
        else:
            es.optimize(objective_fct=self._fit_unsupervised,
                        iterations=self.max_iter,
                        args={'model': model,
                              'params_dict': params,
                              'data_x': x})

        return es.result()
