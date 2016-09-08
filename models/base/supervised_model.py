from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input
import keras.models as kmodels
from models.base.model import Model


class SupervisedModel(Model):

    """ Class representing an abstract Supervised Model.
    """

    def __init__(self,
                 model_name,
                 main_dir,
                 loss_func='mse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 seed=42,
                 verbose=0):

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         loss_func=loss_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self._model_layers = None

    def build_model(self, n_input, n_output=1):

        """ Creates the computational graph for the Supervised Model.
        :param n_input: Number of features.
        :param n_output: number of output values.
        :return: self
        """

        self.logger.info('Building {} model'.format(self.model_name))

        self._input = Input(shape=(n_input,), name='x-input')
        self._model_layers = self._input

        self._create_layers(n_input, n_output)
        
        self._model = kmodels.Model(input=self._input, output=self._model_layers)

        self._model.compile(optimizer=self.opt, loss=self.loss_func)

        self.logger.info('Done building {} model'.format(self.model_name))

    def _create_layers(self, n_input, n_output):
        pass

    def fit(self, x_train, y_train, x_valid, y_valid):

        """ Fit the model to the data.
        :param x_train: Training data. shape(n_samples, n_features)
        :param y_train: Training labels. shape(n_samples, n_classes)
        :param x_valid:
        :param y_valid:
        :return: self
        """

        self.logger.info('Starting {} supervised training...'.format(self.model_name))

        if len(y_train.shape) != 1:
            num_out = y_train.shape[1]
        else:
            self.logger.error('Invalid training labels shape')
            raise Exception("Please convert the labels with one-hot encoding.")

        self.build_model(x_train.shape[1], num_out)

        self._model.fit(x=x_train,
                        y=y_train,
                        batch_size=self.batch_size,
                        nb_epoch=self.num_epochs,
                        verbose=self.verbose,
                        validation_data=(x_valid, y_valid))

        self.logger.info('Done {} supervised training...'.format(self.model_name))

    def predict(self, data):

        """ Predict the labels for the test set.
        :param data: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        preds = self._model.predict(x=data,
                                    batch_size=self.batch_size,
                                    verbose=self.verbose)

        return preds

    def evaluate(self, x, y):

        """ Evaluate the model on (x, y).
        :param x: Input data
        :param y: Target values
        :return:
        """

        cost = self._model.evaluate(x=x,
                                    y=y,
                                    batch_size=self.batch_size,
                                    verbose=self.verbose)

        return cost
