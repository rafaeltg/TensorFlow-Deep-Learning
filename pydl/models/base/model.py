from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os

import keras.optimizers as KOpt
import numpy as np
import tensorflow as tf
from keras.models import load_model, save_model

import pydl.utils.utilities as utils
from pydl.utils.logger import get_logger


class Model:

    """
    Class representing an abstract Model.
    """

    def __init__(self, **kwargs):

        """
        Available parameters:

        :param name: Name of the model, used as filename.
        :param loss_func:
        :param num_epochs:
        :param batch_size:
        :param opt:
        :param learning_rate:
        :param momentum:
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        """

        self.name = kwargs.get('name', self.__class__.__name__)
        self.loss_func = kwargs.get('loss_func', 'mse')
        self.num_epochs = int(kwargs.get('num_epochs', 100))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.opt = str(kwargs.get('opt', 'adam'))
        self.learning_rate = float(kwargs.get('learning_rate', 0.001))

        if self.opt == 'sgd':
            self.momentum = float(kwargs.get('momentum', 0.01))

        self.seed = int(kwargs.get('seed', -1))
        self.verbose = int(kwargs.get('verbose', 0))

        if self.seed >= 0:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

        self.logger = get_logger(self.name, self.verbose)

        self._model = None

        self.validate_params()

    def set_params(self, **params):
        valid_params = self.get_func_params(self.__init__).keys()
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)

        self.validate_params()

    def validate_params(self):
        assert self.num_epochs > 0, 'Invalid number of training epochs'
        assert self.batch_size > 0, 'Invalid batch size'
        assert self.loss_func in utils.valid_loss_functions if isinstance(self.loss_func, str) else True, 'Invalid loss function'
        assert self.opt in utils.valid_optimization_functions, 'Invalid optimizer'
        assert self.learning_rate > 0, 'Invalid learning rate'
        if self.opt == 'sgd':
            assert self.momentum > 0, 'Invalid momentum rate'

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        setattr(self, 'logger', get_logger(self.name, self.verbose))

    def get_model_parameters(self):
        pass

    def save_model(self, path=None, file_name=None):
        assert path is not None, 'Missing output path!'

        if not os.path.exists(path):
            os.makedirs(path)

        if file_name is None:
            file_name = self.name

        w_file = os.path.join(path, file_name + '.h5')
        configs = {
            'model': {
                'class_name': self.__class__.__name__,
                'config': self.get_config()
            },
            'weights': w_file,
        }

        utils.save_json(configs, os.path.join(path, file_name + '.json'))
        save_model(model=self._model, filepath=w_file)

    def load_model(self, model_path, custom_objs=None):
        file_path = model_path
        if os.path.isdir(model_path):
            file_path = os.path.join(model_path, self.name+'.h5')

        assert os.path.isfile(file_path), 'Missing file - %s' % file_path
        self._model = load_model(filepath=file_path, custom_objects=custom_objs)

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        p = self.__getstate__()
        return {k: v for k, v in p.items() if not str(k).startswith('_')}

    def get_loss_func(self):
        return self.loss_func if isinstance(self.loss_func, str) else self.loss_func.__name__

    def get_optimizer(self):

        if self.opt == 'sgd':
            return KOpt.SGD(lr=self.learning_rate, momentum=self.momentum)

        if self.opt == 'rmsprop':
            return KOpt.RMSprop(lr=self.learning_rate)

        if self.opt == 'ada_grad':
            return KOpt.Adagrad(lr=self.learning_rate)

        if self.opt == 'ada_delta':
            return KOpt.Adadelta(lr=self.learning_rate)

        if self.opt == 'adam':
            return KOpt.Adam(lr=self.learning_rate)

        raise Exception('Invalid optimization function')

    @staticmethod
    def get_func_params(f):
        p = inspect.getcallargs(f)
        if 'self' in p:
            del p['self']
        return p

    def copy(self):
        c = self.__new__(self.__class__)
        c.__dict__.update(self.__dict__.copy())
        return c
