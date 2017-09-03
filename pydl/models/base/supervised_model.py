from __future__ import absolute_import

import numpy as np
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from .model import Model
from ..utils import *


class SupervisedModel(Model):

    """
    Class representing an abstract Supervised Model
    """

    def __init__(self, layers=list([]), activation='relu', out_activation='linear', dropout=0,
                 l1_reg=0, l2_reg=0, **kwargs):

        """
        :param layers: 
        :param activation: 
        :param out_activation: 
        :param dropout: 
        :param l1_reg: 
        :param l2_reg: 
        :param kwargs: Model's parameters
        """

        self.layers = layers
        self.activation = expand_arg(self.layers, activation)
        self.out_activation = out_activation
        self.dropout = expand_arg(self.layers, dropout)
        self.l1_reg = expand_arg(self.layers, l1_reg)
        self.l2_reg = expand_arg(self.layers, l2_reg)
        super().__init__(**kwargs)

    def validate_params(self):
        super().validate_params()
        assert self.layers and len(self.layers) > 0, 'Model must have at least one hidden layer'
        assert all([0 <= d <= 1 for d in self.dropout]), 'Invalid dropout value'
        assert all([f in valid_act_functions for f in self.activation]), 'Invalid activation function'
        assert self.out_activation in valid_act_functions, 'Invalid output activation function'
        assert all([x >= 0 for x in self.l1_reg]), 'Invalid l1_reg value'
        assert all([x >= 0 for x in self.l2_reg]), 'Invalid l2_reg value'

    def build_model(self, input_shape, n_output=1, metrics=None):

        """ Creates the computational graph for the Supervised Model.
        :param input_shape:
        :param n_output: number of output values.
        :param metrics:
        :return: self
        """

        self._model = Sequential(name=self.name)

        self._create_layers(input_shape, n_output)

        self._model.compile(optimizer=self.get_optimizer(), loss=self.loss_func, metrics=metrics)

    def _create_layers(self, input_shape, n_output):
        pass

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, valid_split=0.):

        """ Fit the model to the data.
        :param x_train: Training data. shape(n_samples, n_features)
        :param y_train: Training labels. shape(n_samples, n_classes)
        :param x_valid:
        :param y_valid:
        :param valid_split:
        :return: self
        """

        x_train = self._check_x_shape(x_train)
        y_train = self._check_y_shape(y_train)

        self.build_model(x_train.shape, y_train.shape[-1])

        if x_valid is not None and y_valid is not None:
            x_valid = self._check_x_shape(x_valid)
            y_valid = self._check_y_shape(y_valid)
            valid_data = (x_valid, y_valid)
        else:
            valid_data = None

            # By default use 10% of training data for testing
            if self.early_stopping and valid_split == 0.:
                valid_split = 0.1

        self._train_step(x_train, y_train, valid_data, valid_split)

    def _train_step(self, x_train, y_train, valid_data=None, valid_split=0.):
        self._model.fit(x=x_train,
                        y=y_train,
                        batch_size=self.batch_size,
                        epochs=self.nb_epochs,
                        shuffle=False,
                        validation_data=valid_data,
                        validation_split=valid_split,
                        callbacks=self._callbacks,
                        verbose=self.verbose)

    def predict(self, x):

        """ Predict the labels for the test set.
        :param x: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        x = self._check_x_shape(x)

        if self.loss_func == 'binary_crossentropy' or self.loss_func == 'categorical_crossentropy':
            return self._model.predict_classes(x, batch_size=self.batch_size, verbose=self.verbose)

        return self._model.predict(x=x, batch_size=self.batch_size, verbose=self.verbose)

    def predict_proba(self, x):

        """ Predict classes probabilities.
        :param x: Testing data. shape(n_test_samples, n_features)
        :return: probabilities
        """

        if self.loss_func != 'binary_crossentropy' or self.loss_func != 'categorical_crossentropy':
            raise TypeError('Model is not configured to predict classes probabilities. Please, use \
                            "binary_crossentropy" or "categorical_crossentropy" as loss function!')
        
        x = self._check_x_shape(x)

        probs = self._model.predict_proba(x, batch_size=self.batch_size, verbose=self.verbose)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, x, y):

        """ Evaluate the model on (x, y).
        :param x: Input data
        :param y: Target values
        :return:
        """

        x = self._check_x_shape(x)
        y = self._check_y_shape(y)

        loss = self._model.evaluate(x=x, y=y, batch_size=self.batch_size, verbose=self.verbose)

        if isinstance(loss, list):
            return loss[0]
        return loss

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        return self._model.get_weights()

    def get_config(self):
        conf = super().get_config()
        layers = []
        for l in self.layers:
            if isinstance(l, int):
                layers.append(l)
            else:
                layers.append(l.to_json()['model'])
        conf['layers'] = layers
        return conf

    @classmethod
    def from_config(cls, config):
        layers = []
        for l in config['layers']:
            if isinstance(l, dict):
                layers.append(model_from_config(l))
            else:
                layers.append(l)
        config['layers'] = layers
        return cls(**config)

    def _check_x_shape(self, x):
        return x

    def _check_y_shape(self, y):
        y = np.array(y)

        if len(y.shape) == 1:
            if self.loss_func == 'categorical_crossentropy':
                return to_categorical(y)

            return np.reshape(y, (y.shape[0], 1))

        return y
