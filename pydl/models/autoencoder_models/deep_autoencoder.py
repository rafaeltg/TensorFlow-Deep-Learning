import keras.backend as K
import keras.models as kmodels
from keras.layers import Input, Dense
from keras.regularizers import l1l2

from ..base import UnsupervisedModel
from pydl.utils.utilities import expand_arg, valid_act_functions


class DeepAutoencoder(UnsupervisedModel):

    """ Implementation of Deep Autoencoders.
    """

    def __init__(self,
                 name='deep_ae',
                 n_hidden=list([]),
                 enc_activation='relu',
                 dec_activation='linear',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 **kwargs):

        """
        :param layers: List of hidden layers
        """

        super().__init__(name=name,
                         n_hidden=n_hidden,
                         enc_activation=expand_arg(n_hidden, enc_activation),
                         dec_activation=expand_arg(n_hidden, dec_activation),
                         l1_reg=expand_arg(n_hidden, l1_reg),
                         l2_reg=expand_arg(n_hidden, l2_reg),
                         **kwargs)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert all([l > 0 for l in self.n_hidden]), 'Invalid hidden layers!'
        assert all([f in valid_act_functions for f in self.enc_activation]), 'Invalid encoder activation function!'
        assert all([f in valid_act_functions for f in self.dec_activation]), 'Invalid decoder activation function!'
        assert all([x >= 0 for x in self.l1_reg]), 'Invalid l1_reg value!'
        assert all([x >= 0 for x in self.l2_reg]), 'Invalid l2_reg value!'

    def _create_layers(self, input_layer):

        """ Create the encoding and the decoding layers of the deep autoencoder.
        :param input_layer: Input size.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.name))

        encode_layer = input_layer
        for i, l in enumerate(self.n_hidden):
            encode_layer = Dense(output_dim=l,
                                 name='encoder_%d' % i,
                                 activation=self.enc_activation[i],
                                 W_regularizer=l1l2(self.l1_reg[i], self.l2_reg[i]),
                                 b_regularizer=l1l2(self.l1_reg[i], self.l2_reg[i]))(encode_layer)

        self._decode_layer = encode_layer
        for i, l in enumerate(self.n_hidden[-2:-(len(self.n_hidden)+1):-1] + [K.int_shape(input_layer)[1]]):
            self._decode_layer = Dense(output_dim=l,
                                       name='decoder_%d' % i,
                                       activation=self.dec_activation[i])(self._decode_layer)

    def _create_encoder_model(self):

        """ Create the model that maps an input to its encoded representation.
        :return: self
        """

        self.logger.info('Creating {} encoder model'.format(self.name))

        self._encoder = kmodels.Model(input=self._model.layers[0].output,
                                      output=self._model.layers[int(len(self._model.layers)/2)].output)

        self.logger.info('Done creating {} encoder model'.format(self.name))

    def _create_decoder_model(self):

        """ Create the model that maps an encoded input to the original values
        :return: self
        """

        self.logger.info('Creating {} decoder model'.format(self.name))

        dec_idx = int(len(self._model.layers)/2)
        encoded_input = Input(shape=(self._model.layers[dec_idx].output_shape[1],))

        decoder_layer = encoded_input
        for l in self._model.layers[dec_idx+1:]:
            decoder_layer = l(decoder_layer)

        self._decoder = kmodels.Model(input=encoded_input, output=decoder_layer)

        self.logger.info('Done creating {} decoding layer'.format(self.name))

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = {
            'enc': self._encoder.get_weights(),
            'dec': self._decoder.get_weights()
        }

        return params
