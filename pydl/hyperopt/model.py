from ..models import load_model
from ..utils import dump_np_data_set, free_mmap_data_set
from .objective import CVObjectiveFunction
from .optimizer import opt_from_config
import gc


class HyperOptModel(object):

    def __init__(self, hp_space, fit_fn=None, opt='cmaes', opt_args=dict([])):
        self._fit_fn = fit_fn if fit_fn else CVObjectiveFunction()
        self._opt = opt_from_config(algo=opt, **opt_args)
        self._hp_space = hp_space
        self._best_model = None
        self._best_config = None

    @property
    def best_config(self):
        return self._best_config

    @property
    def best_model(self):
        return self._best_model

    def fit(self, x, y=None, retrain=False, max_threads=1, in_memory=True):

        try:
            if in_memory:
                x, y = dump_np_data_set(x, y)
                gc.collect()

            res = self._opt.optimize(
                x0=[0] * self._hp_space.size,
                obj_func=self._fit_fn.obj_fn,
                args=(self._hp_space, x, y) + self._fit_fn.args,
                max_threads=max_threads)

            self._best_config = self._hp_space.get_value(res[0])
            self._best_model = load_model(self._best_config)

            if retrain:
                if y is not None:
                    self._best_model.fit(x, y)
                else:
                    self._best_model.fit(x)

        finally:
            if in_memory:
                free_mmap_data_set(x, y)

        return {
            'opt_result': res,
            'best_model_config': self._best_config
        }

    def save_model(self, path=None, file_name=None):
        self._best_model.save_model(path, file_name)
