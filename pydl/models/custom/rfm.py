from __future__ import absolute_import

import os
import keras.models as k_models
from ..base import Model
from ..utils import *


class RFM:

    """
    Residual Forecasting Model
    """

    def __init__(self, name='rfm', fm=None, rm=None):

        """
        :param fm: Forecast model
        :param rm: Residual Forecasting Model
        """

        self.name = name
        self._fm = fm
        self._rm = rm
        self._validate_params()

    def _validate_params(self):
        assert self._fm is not None, 'Forecast Model cannot be empty'
        assert isinstance(self._fm, Model), 'Forecast Model must be of class pydl.Model'
        assert self._rm is not None, 'Residual Forecasting Model cannot be empty'
        assert isinstance(self._rm, Model), 'Residual Forecasting Model must be of class pydl.Model'

    def build_model(self, input_shape, n_output=1, metrics=None):

        """ Creates the computational graph for the Supervised Models.
        :param input_shape:
        :param n_output: number of output values.
        :param metrics:
        :return: self
        """

        self._fm.build_model(input_shape, n_output, metrics)
        self._rm.build_model(input_shape, n_output, metrics)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, valid_split=0.):

        """ Fit the model to the data.
        :param x_train: Training data. shape(n_samples, n_features)
        :param y_train: Training labels. shape(n_samples, n_classes)
        :param x_valid:
        :param y_valid:
        :param valid_split:
        :return: self
        """

        self.build_model(x_train.shape, y_train.shape[-1])

        self._fm.fit(x_train, y_train, x_valid, y_valid, valid_split)

        y_train_pred = self._fm.predict(x_train)

        e = y_train - y_train_pred

        self._rm.fit(x_train=x_train, y_train=e)

    def predict(self, x):

        """ Predict the labels for the test set.
        :param x: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        y_pred = self._fm.predict(x)
        e_pred = self._rm.predict(x)
        return y_pred + e_pred

    def get_config(self):
        return {
            'name': self.name,
            'fm': self._fm.to_json(),
            'rm': self._rm.to_json(),
        }

    @classmethod
    def from_config(cls, config):
        return cls(name=config['name'] if 'name' in config else 'rfm',
                   fm=load_model(config['fm']),
                   em=load_model(config['rm']))

    def to_json(self):
        return {
            'model': {
                'class_name': self.__class__.__name__,
                'config': self.get_config()
            }
        }

    def is_built(self):
        return self._fm.is_built() and self._rm.is_built()

    def save(self, dir=None, file_name=None):
        assert os.path.exists(dir), 'Directory does not exist'

        if file_name is None:
            file_name = self.name

        cfg = self.to_json()

        if self.is_built():
            cfg['weights'] = {
                'fm': os.path.join(dir, file_name + '_fm.h5'),
                'rm': os.path.join(dir, file_name + '_rm.h5')
            }

            k_models.save_model(getattr(self._fm, '_model'), cfg['weights']['fm'])
            k_models.save_model(getattr(self._rm, '_model'), cfg['weights']['rm'])

        save_json(cfg, os.path.join(dir, file_name+'.json'))

    def load_model(self, model_paths, custom_objs=None):
        assert 'fm' in model_paths
        assert 'rm' in model_paths

        self._fm.load_model(model_path=model_paths['fm'], custom_objs=custom_objs)
        self._rm.load_model(model_path=model_paths['rm'], custom_objs=custom_objs)
