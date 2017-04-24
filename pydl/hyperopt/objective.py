from ..models.utils import load_model
from ..model_selection import CV


class CVObjectiveFunction:

    def __init__(self, scoring=None, cv_method='split', **kwargs):
        self._cv = CV(method=cv_method, **kwargs)
        self._scoring = scoring

    @property
    def args(self):
        return tuple([self._cv, self._scoring])

    def obj_fn(self, x, *args):
        hp_space = args[0]
        X = args[1]
        Y = args[2]

        m = load_model(hp_space.get_value(x))
        res = self._cv.run(model=m, x=X, y=Y, scoring=self._scoring)
        s = self._cv.get_scorer_name(self._scoring) if self._scoring is not None else m.get_loss_func()
        return res[s]['mean']

    def __call__(self, x, *args):
        self.obj_fn(x, *args)
