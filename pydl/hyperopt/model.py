from .optimizer import CMAESOptimizer
from ..model_selection import KFold
from ..utils.utilities import load_model


class HyperOptModel(object):

    def __init__(self, hp_space, cv=None, opt=None):
        self._cv = cv if cv else KFold()
        self._opt = opt if opt else CMAESOptimizer()
        self._hp_space = hp_space
        self._best_model = None

    @property
    def cv(self):
        return self._cv

    @cv.setter
    def cv(self, value):
        self._cv = value

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

        args = (self._hp_space, x, y, self._cv)
        res = self._opt.optimize(x0=[0]*self.hp_space.size, obj_func=self._opt_obj_fn, args=args)
        self._best_model = load_model(self._hp_space.get_config(res[0]))

        if retrain:
            if y:
                self._best_model.fit(x, y)
            else:
                self._best_model.fit(x)

        return {
            'opt_result': res,
            'best_model_config': self._hp_space.get_config(res[0])
        }

    @staticmethod
    def _opt_obj_fn(x, hp_space, data_x, data_y, cv):
        m = load_model(hp_space.get_value(x))
        res = cv.run(model=m, x=data_x, y=data_y, max_thread=1)
        return res[m.get_loss_func()]['mean']