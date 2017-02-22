from ..utils.utilities import load_model
from ..model_selection import CV


class ObjectiveFunction:

    def __init__(self):
        self._args = ()

    @property
    def args(self):
        return self._args

    @property
    def obj_fn(self):
        return self._obj_fn

    @staticmethod
    def _obj_fn(x, hp_space, data_x, data_y, *args):
        pass


class CVObjectiveFunction(ObjectiveFunction):

    def __init__(self, cv=None, **kwargs):
        super().__init__()
        self._args += tuple([CV(cv, **kwargs) if cv is not None else CV(method='split')])

    @staticmethod
    def _obj_fn(x, hp_space, data_x, data_y, *args):
        cv = args[0]
        m = load_model(hp_space.get_value(x))
        res = cv.run(model=m, x=data_x, y=data_y, max_thread=1)
        return res[m.get_loss_func()]['mean']