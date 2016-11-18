from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.models as kmodels
from keras.layers import Input
from pydl.models.base.model import Model


class UnsupervisedModel(Model):

    """ Class representing an abstract Unsupervised Model.
    """

    def __init__(self,
                 name,
                 loss_func='mse',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 seed=42,
                 verbose=0):

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

        # Model input data
        self._input = None

        self._decode_layer = None

        self._encoder = None
        self._decoder = None

    def build_model(self, n_input):

        """ Creates the computational graph for the Unsupervised Model.
        :param n_input: Number of features.
        :return: self
        """

        self.logger.info('Building {} model'.format(self.name))

        self._input = Input(shape=(n_input,), name='x-input')

        self._create_layers(self._input)
        self._model = kmodels.Model(input=self._input, output=self._decode_layer)

        self._create_encoder_model()
        self._create_decoder_model()

        opt = self.get_optimizer(opt_func=self.opt_func,
                                 learning_rate=self.learning_rate,
                                 momentum=self.momentum)

        self._model.compile(optimizer=opt, loss=self.loss_func)

        self.logger.info('Done building {} model'.format(self.name))

    def _create_layers(self, n_input):
        pass

    def _create_encoder_model(self):
        pass

    def _create_decoder_model(self):
        pass

    def fit(self, x_train, x_valid=None):

        """
        :param x_train: Training data. shape(n_samples, n_features)
        :param x_valid: Validation data. shape(n_samples, n_features)
        :return:
        """

        self.logger.info('Starting {} unsupervised training...'.format(self.name))

        self.build_model(x_train.shape[1])

        self._model.fit(x=x_train,
                        y=x_train,
                        nb_epoch=self.num_epochs,
                        batch_size=self.batch_size,
                        shuffle=False,
                        validation_data=(x_valid, x_valid) if x_valid else None,
                        verbose=self.verbose)

        self.logger.info('Done {} unsupervised training...'.format(self.name))

    def transform(self, data):

        """ Transform data according to the model.
        :param data: Data to transform
        :return: transformed data
        """

        self.logger.info('Transforming data...')

        encoded_data = self._encoder.predict(x=data, 
                                             verbose=self.verbose)

        return encoded_data

    def reconstruct(self, encoded_data):

        """
        :param encoded_data:
        :return:
        """

        rec_data = self._decoder.predict(x=encoded_data, 
                                         verbose=self.verbose)

        return rec_data

    def score(self, data):

        """ Compute the total reconstruction loss.
        :param data: Input data
        :return: reconstruction cost
        """

        self.logger.info('Evaluating reconstruction loss...')

        loss = self._model.evaluate(x=data,
                                    y=data,
                                    batch_size=self.batch_size,
                                    verbose=self.verbose)

        if type(loss) is list:
            return loss[0]
        return loss

    def load_model(self, model_path):
        super().load_model(model_path)
        self._create_encoder_model()
        self._create_decoder_model()
