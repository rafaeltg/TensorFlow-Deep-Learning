import sys
import os
import tempfile
import shutil
from joblib import load, dump
from ..models import load_model
from .objective import CVObjectiveFunction
from .optimizer import opt_from_config


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

    @staticmethod
    def _dump(var, file_path):
        mmap_file = os.path.join(file_path)
        if os.path.exists(mmap_file):
            os.unlink(mmap_file)
        dump(var, mmap_file)
        return load(mmap_file, mmap_mode='r')

    def fit(self, x, y=None, retrain=False, max_threads=1):

        tmp_folder = tempfile.mkdtemp(dir='/dev/shm')

        try:
            x = self._dump(x, os.path.join(tmp_folder, 'x.mmap'))
            y = self._dump(y, os.path.join(tmp_folder, 'y.mmap')) if y is not None else None
            print(sys.getsizeof(self._hp_space))

            res = self._opt.optimize(obj_func=self._fit_fn,
                                     args=(self._hp_space, x, y),
                                     max_threads=max_threads)

            self._best_config = self._hp_space.get(res[0])
            self._best_model = load_model(self._best_config)

            if retrain:
                if y is not None:
                    self._best_model.fit(x, y)
                else:
                    self._best_model.fit(x)

        finally:
            try:
                shutil.rmtree(tmp_folder)
            except:
                print("Failed to delete: " + tmp_folder)

        return {
            'opt_result': res,
            'best_model_config': self._best_config
        }
