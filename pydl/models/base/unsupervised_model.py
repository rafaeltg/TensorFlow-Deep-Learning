from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .model import Model
import keras.models as kmodels
from keras.layers import Input


class UnsupervisedModel(Model):

    """ Class representing an abstract Unsupervised Model.
    """

    def __init__(self,
                 n_hidden=None,
                 enc_activation='relu',
                 dec_activation='linear',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 **kwargs):

        self.n_hidden = n_hidden
        self.enc_activation = enc_activation
        self.dec_activation = dec_activation
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        super().__init__(**kwargs)

        self._input = None
        self._decode_layer = None
        self._encoder = None
        self._decoder = None

    def build_model(self, input_shape):

        """ Creates the computational graph for the Unsupervised Model.
        """

        self.logger.info('Building {} model'.format(self.name))

        self._input = Input(shape=input_shape[1:], name='x-input')

        self._create_layers(self._input)
        self._model = kmodels.Model(input=self._input, output=self._decode_layer)

        self._create_encoder_model()
        self._create_decoder_model()

        self._model.compile(optimizer=self.get_optimizer(), loss=self.loss_func)

        self.logger.info('Done building {} model'.format(self.name))

    def _create_layers(self, input_layer):
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

        self.build_model(x_train.shape)

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
        return self._encoder.predict(x=data, verbose=self.verbose)

    def reconstruct(self, encoded_data):

        """
        :param encoded_data:
        :return:
        """

        self.logger.info('Reconstructing data...')
        return self._decoder.predict(x=encoded_data, verbose=self.verbose)

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

    def load_model(self, model_path, custom_objs=None):
        super().load_model(model_path, custom_objs)
        self._create_encoder_model()
        self._create_decoder_model()
