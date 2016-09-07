from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.optimizers as KOpt
import numpy as np
import tensorflow as tf

import utils.utilities as utils
from utils.logger import Logger


class Model:

    """ Class representing an abstract Model.
    """

    def __init__(self,
                 model_name,
                 main_dir,
                 loss_func='mean_squared_error',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.1,
                 seed=-1,
                 verbose=0):

        """
        :param model_name: Name of the model, used as filename.
        :param main_dir: Main directory to put the stored_models, data and summary directories.
        :param loss_func:
        :param num_epochs:
        :param batch_size:
        :param opt:
        :param learning_rate:
        :param momentum:
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        """

        # Validations
        assert model_name is not ''
        assert main_dir is not ''
        assert num_epochs > 0
        assert batch_size > 0
        assert loss_func in utils.valid_loss_functions
        assert opt in utils.valid_optimization_functions
        assert learning_rate > 0
        assert momentum > 0 if opt == 'sgd' else True

        self.model_name = model_name
        self.main_dir = main_dir

        self._model = None
        self._input = None # Model input data
        self.loss_func = loss_func

        # Training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Optimization function
        self.opt = self.get_optimizer(opt_func=opt, learning_rate=learning_rate, momentum=momentum)

        if seed >= 0:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        self.verbose = verbose

        # Create the logger
        self.logger = Logger(model_name, verbose)

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

    def get_model_parameters(self):
        pass
