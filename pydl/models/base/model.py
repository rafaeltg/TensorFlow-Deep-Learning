from __future__ import absolute_import

import os
import inspect
import numpy as np
import keras.models as k_models
import keras.optimizers as k_opt
from keras.callbacks import EarlyStopping
from ..utils import *


class Model:

    """
    Class representing an abstract Model.
    """

    def __init__(self, name='', loss_func='mse', nb_epochs=100, batch_size=32, opt='adam',
                 learning_rate=0.001, momentum=0.01, early_stopping=False, patient=2,
                 min_delta=1e-4, seed=-1, verbose=0):

        """
        Available parameters:

        :param name: Name of the model.
        :param loss_func:
        :param nb_epochs:
        :param batch_size:
        :param opt:
        :param learning_rate:
        :param momentum:
        :param early_stopping: 
        :param patient: number of epochs with no improvement after which training will be stopped.
        :param min_delta:
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        """

        self.name = name if name != '' else self.__class__.__name__
        self.loss_func = loss_func
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.opt = opt
        self.learning_rate = learning_rate

        if self.opt == 'sgd':
            self.momentum = momentum

        # Early stopping callback
        self.early_stopping = early_stopping
        self.patient = patient
        self.min_delta = min_delta
        self._callbacks = [EarlyStopping(min_delta=self.min_delta, patience=self.patient)] if self.early_stopping else []

        self.seed = seed
        if self.seed >= 0:
            np.random.seed(self.seed)

        self.verbose = verbose

        self._model = None

        self.validate_params()

    def set_params(self, **params):
        valid_params = self.get_func_params(self.__init__).keys()
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)

        self.validate_params()

    def validate_params(self):
        assert self.nb_epochs > 0, 'Invalid number of training epochs'
        assert self.batch_size > 0, 'Invalid batch size'
        assert self.loss_func in valid_loss_functions if isinstance(self.loss_func, str) else True, 'Invalid loss function'
        assert self.opt in valid_opt_functions, 'Invalid optimizer'
        assert self.learning_rate > 0, 'Invalid learning rate'
        if self.opt == 'sgd':
            assert self.momentum > 0, 'Invalid momentum rate'

        assert isinstance(self.early_stopping, bool), 'Invalid early_stopping value'
        if self.early_stopping:
            assert self.min_delta > 0, 'Invalid min_delta value'
            assert self.patient > 0, 'Invalid patient value'

    def get_model_parameters(self):
        pass

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        conf = {k: v for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)}
        return conf

    def get_loss_func(self):
        return self.loss_func if isinstance(self.loss_func, str) else self.loss_func.__name__

    def get_optimizer(self):

        if self.opt == 'sgd':
            return k_opt.SGD(lr=self.learning_rate, momentum=self.momentum)

        if self.opt == 'rmsprop':
            return k_opt.RMSprop(lr=self.learning_rate)

        if self.opt == 'adagrad':
            return k_opt.Adagrad(lr=self.learning_rate)

        if self.opt == 'adadelta':
            return k_opt.Adadelta(lr=self.learning_rate)

        if self.opt == 'adam':
            return k_opt.Adam(lr=self.learning_rate)

        raise Exception('Invalid optimization function - %s' % self.opt)

    def get_func_params(self, f):
        p = inspect.getcallargs(f)
        if 'self' in p:
            del p['self']
        return p

    def copy(self):
        c = self.__new__(self.__class__)
        c.__dict__.update(self.__dict__.copy())
        return c

    def to_json(self):
        return {
            'model': {
                'class_name': self.__class__.__name__,
                'config': self.get_config()
            }
        }

    def load_model(self, model_path, custom_objs=None):
        file_path = model_path
        if os.path.isdir(model_path):
            file_path = os.path.join(model_path, self.name+'.h5')

        assert os.path.isfile(file_path), 'Missing file - %s' % file_path
        self._model = k_models.load_model(filepath=file_path, custom_objects=custom_objs)

    def is_built(self):
        return self._model is not None
