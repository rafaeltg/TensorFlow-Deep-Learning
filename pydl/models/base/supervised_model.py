from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential

import pydl.utils.utilities as utils
from pydl.models.base.model import Model
from keras.utils.layer_utils import layer_from_config
from keras.layers import Dense
from keras.regularizers import l1l2


class SupervisedModel(Model):

    """ Class representing an abstract Supervised Model.
    """

    def __init__(self,
                 name,
                 layers=None,
                 dec_act_func='linear',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 loss_func='mse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 verbose=0,
                 seed=42):

        self.layers = layers
        self.dec_act_func = dec_act_func

        super().__init__(name=name,
                         loss_func=loss_func,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

    def validate_params(self):
        super().validate_params()
        assert len(self.layers) > 0, 'Model must have at least one hidden layer'
        assert self.dec_act_func in utils.valid_act_functions, 'Invalid decoder layer activation function'

    def build_model(self, input_shape, n_output=1):

        """ Creates the computational graph for the Supervised Model.
        :param input_shape:
        :param n_output: number of output values.
        :return: self
        """

        self.logger.info('Building {} model'.format(self.name))

        layers = self._create_layers(input_shape, n_output)

        self._model = Sequential(layers=layers, name=self.name)

        self._model.summary()
        
        opt = self.get_optimizer(opt_func=self.opt,
                                 learning_rate=self.learning_rate,
                                 momentum=self.momentum)

        self._model.compile(optimizer=opt, loss=self.loss_func)

        self.logger.info('Done building {} model'.format(self.name))

    def _create_layers(self, input_shape, n_output):

        if not hasattr(self.layers[0], 'batch_input_shape'):
            self.layers[0].create_input_layer(input_shape[1:])

        self.add_output_layer(n_output)

        return self.layers

    def add_output_layer(self, n_out):
        self.layers.append(Dense(output_dim=n_out,
                                 activation=self.dec_act_func,
                                 W_regularizer=l1l2(self.l1_reg, self.l2_reg),
                                 b_regularizer=l1l2(self.l1_reg, self.l2_reg)))

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):

        """ Fit the model to the data.
        :param x_train: Training data. shape(n_samples, n_features)
        :param y_train: Training labels. shape(n_samples, n_classes)
        :param x_valid:
        :param y_valid:
        :return: self
        """

        self.logger.info('Starting {} supervised training...'.format(self.name))

        if len(y_train.shape) != 1:
            num_out = y_train.shape[1]
        else:
            self.logger.error('Invalid training labels shape')
            raise Exception("Please convert the labels with one-hot encoding.")

        self.build_model(x_train.shape, num_out)

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

    def predict(self, data):

        """ Predict the labels for the test set.
        :param data: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        preds = self._model.predict(x=data,
                                    batch_size=self.batch_size,
                                    verbose=self.verbose)

        return preds

    def score(self, x, y):

        """ Evaluate the model on (x, y).
        :param x: Input data
        :param y: Target values
        :return:
        """

        loss = self._model.evaluate(x=x,
                                    y=y,
                                    batch_size=self.batch_size,
                                    verbose=self.verbose)

        if type(loss) is list:
            return loss[0]
        return loss

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        return self._model.get_weights()

    def get_config(self):
        conf = super().get_config()
        conf['dec_act_func'] = self.dec_act_func
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
        layers = []
        for l in config['layers']:
            if isinstance(l, Model):
                utils.build_model()
            layers.append(layer_from_config(l))
        config['layers'] = layers
        return cls(**config)
