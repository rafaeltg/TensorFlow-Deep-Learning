import numpy as np
from .model import Model
from pydl.utils.utilities import layers_from_config
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical


class SupervisedModel(Model):

    """ Class representing an abstract Supervised Model.
    """

    def __init__(self, layers=None, **kwargs):
        self.layers = layers
        super().__init__(**kwargs)

    def validate_params(self):
        super().validate_params()
        assert self.layers and len(self.layers) > 0, 'Model must have at least one hidden layer'

    def build_model(self, input_shape, n_output=1, metrics=None):

        """ Creates the computational graph for the Supervised Model.
        :param input_shape:
        :param n_output: number of output values.
        :param metrics:
        :return: self
        """

        self.logger.info('Building {} model'.format(self.name))

        layers = self._create_layers(input_shape, n_output)

        self._model = Sequential(layers=layers, name=self.name)

        self._model.compile(optimizer=self.get_optimizer(), loss=self.loss_func, metrics=metrics)

        self.logger.info('Done building {} model'.format(self.name))

    def _create_layers(self, input_shape, n_output):

        if not hasattr(self.layers[0], 'batch_input_shape'):
            self.layers[0].create_input_layer(input_shape[1:])

        self._check_output_layer(n_output)

        return self.layers

    def _check_output_layer(self, n_output):
        for l in self.layers[::-1]:
            if isinstance(l, Dense):
                if l.output_dim != n_output:
                    self.logger.info('Missing output layer! Creating default layer based on output shape...')
                    self.layers.append(Dense(output_dim=n_output))
                break

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):

        """ Fit the model to the data.
        :param x_train: Training data. shape(n_samples, n_features)
        :param y_train: Training labels. shape(n_samples, n_classes)
        :param x_valid:
        :param y_valid:
        :return: self
        """

        self.logger.info('Starting {} supervised training...'.format(self.name))

        y_train = self._check_y_shape(y_train)
        y_valid = self._check_y_shape(y_valid) if y_valid else None

        self.build_model(x_train.shape, y_train.shape[-1])

        self._train_step(x_train, y_train, x_valid, y_valid)

        self.logger.info('Done {} supervised training...'.format(self.name))

    def _train_step(self, x_train, y_train, x_valid=None, y_valid=None):
        self._model.fit(x=x_train,
                        y=y_train,
                        batch_size=self.batch_size,
                        nb_epoch=self.num_epochs,
                        verbose=self.verbose,
                        shuffle=False,
                        validation_data=(x_valid, y_valid) if x_valid and y_valid else None)

    def predict(self, x, class_probs=False):

        """ Predict the labels for the test set.
        :param x: Testing data. shape(n_test_samples, n_features)
        :param class_probs: In case of a classifier, it indicates whether it should returns
            the class label or class probability for the given data.
        :return: labels
        """

        if self.loss_func == 'binary_crossentropy' or self.loss_func == 'categorical_crossentropy':
            if not class_probs:
                return self._model.predict_classes(x, batch_size=self.batch_size, verbose=self.verbose)

            else:
                probs = self._model.predict_proba(x, batch_size=self.batch_size, verbose=self.verbose)

                # check if binary classification
                if probs.shape[1] == 1:
                    # first column is probability of class 0 and second is of class 1
                    probs = np.hstack([1 - probs, probs])
                return probs

        return self._model.predict(x=x, batch_size=self.batch_size, verbose=self.verbose)

    def score(self, x, y):

        """ Evaluate the model on (x, y).
        :param x: Input data
        :param y: Target values
        :return:
        """

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
            layers.append({
                'class_name': l.__class__.__name__,
                'config': l.get_config(),
            })
        conf['layers'] = layers
        return conf

    @classmethod
    def from_config(cls, config):
        config['layers'] = layers_from_config(config['layers'])
        return cls(**config)

    def _check_y_shape(self, y):
        y = np.array(y)

        if len(y.shape) == 1:
            if self.loss_func == 'categorical_crossentropy':
                return to_categorical(y)

            return np.reshape(y, (y.shape[0], 1))

        return y
