import copy

import keras.backend as K
import keras.models as kmodels
from keras.layers import Input, Dense

from ..base import UnsupervisedModel
from pydl.utils.utilities import layers_from_config


class DeepAutoencoder(UnsupervisedModel):

    """ Implementation of Deep Autoencoders.
    """

    def __init__(self,
                 name='deep_ae',
                 layers=None,
                 **kwargs):

        """
        :param layers: List of hidden layers
        """

        self.layers = layers

        super().__init__(name=name, **kwargs)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        assert self.layers is not None, 'Missing hidden layers!'

    def _create_layers(self, input_layer):

        """ Create the encoding and the decoding layers of the deep autoencoder.
        :param input_layer: Input size.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.name))

        encode_layer = input_layer
        for i, l in enumerate(copy.deepcopy(self.layers)):
            l.name = 'encoder_%d' % i
            encode_layer = l(encode_layer)

        self._decode_layer = encode_layer
        for i, l in enumerate(self.layers[-2:-(len(self.layers)+1):-1]):
            l.name = 'decoder_%d' % i
            self._decode_layer = l(self._decode_layer)

        self._decode_layer = Dense(name='decoder_%d' % (len(self.layers)-1),
                                   output_dim=K.int_shape(input_layer)[1])(self._decode_layer)

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
