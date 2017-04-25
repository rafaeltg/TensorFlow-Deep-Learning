from ..models.utils import load_model
from ..model_selection import CV


class CVObjectiveFunction:

    def __init__(self, scoring=None, cv_method='split', **kwargs):
        self._cv = CV(method=cv_method, **kwargs)
        self._scoring = scoring

    @property
    def args(self):
        return tuple([self._cv, self._scoring])

    @staticmethod
    def obj_fn(x, *args):
        hp_space = args[0]
        X = args[1]
        Y = args[2]
        cv = args[3]
        scoring = args[4]

        m = load_model(hp_space.get_value(x))
        res = cv.run(model=m, x=X, y=Y, scoring=scoring)
        s = cv.get_scorer_name(scoring) if scoring is not None else m.get_loss_func()
        return res[s]['mean']

    def __call__(self, x, *args):
        return self.obj_fn(x, *args, self._cv, self._scoring)
