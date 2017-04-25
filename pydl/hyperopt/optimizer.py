import cma
from joblib import Parallel, delayed


class Optimizer:

    def __init__(self, max_iter=25, verbose=-9, **kwargs):
        assert max_iter > 0, 'max_iter must be greater than zero!'
        self.max_iter = max_iter
        self.verbose = verbose

    def optimize(self, x0, obj_func, args=(), max_threads=1):
        pass


class CMAESOptimizer(Optimizer):

    def __init__(self, max_iter=25, verbose=-9, **kwargs):
        super().__init__(max_iter=max_iter, verbose=verbose, **kwargs)

        self.pop_size = int(kwargs.get('pop_size', 20)) if 'pop_size' in kwargs else 20
        assert self.pop_size > 0, 'pop_size must be greater than zero'

        self.sigma0 = float(kwargs.get('sigma0', 0.5)) if 'sigma0' in kwargs else 0.5
        assert self.sigma0 > 0, 'sigma0 must be greater than zero'

    def optimize(self, x0, obj_func, args=(), max_threads=1):
        assert max_threads > 0, 'Invalid number of threads!'
        assert len(args) >= 3, 'args must contain at least hp_space, data_X and data_Y'

        # TODO: 'AdaptSigma' option
        es = cma.CMAEvolutionStrategy(x0=x0,
                                      sigma0=self.sigma0,
                                      inopts={
                                          'maxiter': self.max_iter,
                                          'bounds': [0, 1],
                                          'popsize': self.pop_size,
                                          'verbose': self.verbose
                                      })

        if max_threads == 1:
            es.optimize(objective_fct=obj_func, args=args)
        else:
            max_threads = min(self.pop_size, max_threads)
            while not es.stop():
                X = es.ask()

                with Parallel(n_jobs=max_threads) as parallel:
                    f_values = parallel(delayed(function=obj_func, check_pickle=False)(_x, *args) for _x in X)

                es.tell(X, f_values)
                es.disp()
                es.logger.add()

        return es.result()


def opt_from_config(algo, **kwargs):
    if algo == 'cmaes':
        return CMAESOptimizer(**kwargs)

    raise TypeError('Invalid optimizer algorithm (%s)' % algo)
