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
from pydl.utils.logger import Logger


class Model:

    """
    Class representing an abstract Model.
    """

    def __init__(self,
                 name='',
                 loss_func='mse',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.1,
                 seed=-1,
                 verbose=0):

        """
        :param name: Name of the model, used as filename.
        :param loss_func:
        :param l1_reg: L1 weight regularization penalty, also known as LASSO.
        :param l2_reg: L2 weight regularization penalty, also known as weight decay, or Ridge.
        :param num_epochs:
        :param batch_size:
        :param opt:
        :param learning_rate:
        :param momentum:
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        """

        self.name = name
        self.loss_func = loss_func
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.opt = opt
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.seed = seed
        self.verbose = verbose

        self.validate_params()

        self._model = None

        if seed >= 0:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        # Create the logger
        self.logger = Logger(name, verbose)

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
        assert self.l1_reg >= 0
        assert self.l2_reg >= 0
        assert self.opt in utils.valid_optimization_functions, 'Invalid optimizer'
        assert self.learning_rate > 0, 'Invalid learning rate'
        assert self.momentum > 0 if self.opt == 'sgd' else True, 'Invalid momentum rate'

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        logger = getattr(self, 'logger', None)
        if logger is None:
            setattr(self, 'logger', Logger(self.name, self.verbose))

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        p = self.__getstate__()
        return {k: p[k] for k in self.get_func_params(self.__init__).keys()}

    def load_weights(self, model_path):
        file_path = model_path
        if os.path.isdir(model_path):
            file_path = os.path.join(model_path, self.name+'.h5')

        assert os.path.isfile(file_path), 'Missing file - %s' % file_path
        self._load_model(file_path)

    def _load_model(self, config_file):
        self._model = load_model(filepath=config_file)

    def get_loss_func(self):
        return self.loss_func if isinstance(self.loss_func, str) else self.loss_func.__name__

    @staticmethod
    def get_optimizer(opt_func, learning_rate, momentum):

        if opt_func == 'sgd':
            return KOpt.SGD(lr=learning_rate, momentum=momentum)

        if opt_func == 'rmsprop':
            return KOpt.RMSprop(lr=learning_rate)

        if opt_func == 'ada_grad':
            return KOpt.Adagrad(lr=learning_rate)

        if opt_func == 'ada_delta':
            return KOpt.Adadelta(lr=learning_rate)

        if opt_func == 'adam':
            return KOpt.Adam(lr=learning_rate)

        raise Exception('Invalid optimization function')

    @staticmethod
    def get_func_params(f):
        p = inspect.getcallargs(f)
        if 'self' in p:
            del p['self']
        return p
