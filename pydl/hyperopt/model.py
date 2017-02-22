from .optimizer import CMAESOptimizer
from .objective import CVObjectiveFunction
from ..utils.utilities import load_model


class HyperOptModel(object):

    def __init__(self, hp_space, fit_fn=None, opt=None):
        self._fit_fn = fit_fn if fit_fn else CVObjectiveFunction()
        self._opt = opt if opt else CMAESOptimizer()
        self._hp_space = hp_space
        self._best_model = None

    @property
    def fit_fn(self):
        return self._fit_fn

    @fit_fn.setter
    def fit_fn(self, value):
        self._fit_fn = value

    @property
    def opt(self):
        return self._opt

    @opt.setter
    def opt(self, value):
        self._opt = value

    @property
    def hp_space(self):
        return self._hp_space

    @hp_space.setter
    def hp_space(self, value):
        self._hp_space = value

    @property
    def best_model(self):
        return self._best_model

    def fit(self, x, y=None, retrain=False):

        args = (self._hp_space, x, y) + self._fit_fn.args
        res = self._opt.optimize(x0=[0]*self.hp_space.size, obj_func=self._fit_fn.obj_fn, args=args)
        best_config = self._hp_space.get_value(res[0])
        self._best_model = load_model(best_config)

        if retrain:
            if y:
                self._best_model.fit(x, y)
            else:
                self._best_model.fit(x)

        return {
            'opt_result': res,
            'best_model_config': best_config
        }