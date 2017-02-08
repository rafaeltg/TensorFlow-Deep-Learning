from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.models as kmodels
from keras import backend as K
from keras import objectives
from keras.layers import Input, Dense, Lambda
from keras.models import load_model

from pydl.utils.utilities import valid_act_functions
from ..base import UnsupervisedModel


class VariationalAutoencoder(UnsupervisedModel):

    """ Implementation of a Variational Autoencoder.
    """

    def __init__(self,
                 name='vae',
                 n_latent=10,
                 n_hidden=64,
                 enc_act_func='relu',
                 dec_act_func='relu',
                 num_epochs=10,
                 batch_size=100,
                 opt='rmsprop',
                 learning_rate=0.01,
                 momentum=0.5,
                 verbose=0,
                 seed=-1):

        """
        :param n_latent: number of units in the latent layer
        :param n_hidden: number of hidden units
        :param enc_act_func: Activation function for the encoder.
        :param dec_act_func: Activation function for the decoder.
        :param num_epochs: Number of epochs for training
        :param batch_size: Size of each mini-batch
        :param opt: Which optimizer to use.
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func

        super().__init__(name=name,
                         loss_func=self._vae_loss,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.n_inputs = None

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert self.n_latent > 0
        assert self.n_hidden > 0
        assert self.enc_act_func in valid_act_functions
        assert self.dec_act_func in valid_act_functions

    def _create_layers(self, input_layer):

        """ Create the encoding and the decoding layers of the variational autoencoder.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.name))

        self.n_inputs = K.int_shape(input_layer)[1]

        # Encode layers
        encode_layer = Dense(output_dim=self.n_hidden,
                             activation=self.enc_act_func)(input_layer)

        z_mean = Dense(name='z_mean', output_dim=self.n_latent)(encode_layer)
        z_log_var = Dense(name='z_log_var', output_dim=self.n_latent)(encode_layer)

        z = Lambda(self._sampling, output_shape=(self.n_latent,))([z_mean, z_log_var])

        # Decode layers
        self._decode_layer = Dense(output_dim=self.n_hidden,
                                   activation=self.dec_act_func)(z)

        self._decode_layer = Dense(output_dim=self.n_inputs, activation='linear')(self._decode_layer)

    def _create_encoder_model(self):

        """ Create the encoding layer of the variational autoencoder.
        :return: self
        """

        self.logger.info('Creating {} encoder model'.format(self.name))

        # This model maps an input to its encoded representation
        self._encoder = kmodels.Model(input=self._model.layers[0].inbound_nodes[0].output_tensors,
                                      output=self._model.get_layer('z_mean').inbound_nodes[0].output_tensors)

        self.logger.info('Done creating {} encoder model'.format(self.name))

    def _create_decoder_model(self):

        """ Create the decoding layers of the variational autoencoder.
        :return: self
        """

        self.logger.info('Creating {} decoder model'.format(self.name))

        encoded_input = Input(shape=(self.n_latent,))

        decoder_layer = self._model.layers[-2](encoded_input)
        decoder_layer = self._model.layers[-1](decoder_layer)

        # create the decoder model
        self._decoder = kmodels.Model(input=encoded_input, output=decoder_layer)

        self.logger.info('Done creating {} decoding layer'.format(self.name))

    @staticmethod
    def _sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_log_var), mean=0., std=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        z_mean = self._model.get_layer('z_mean').inbound_nodes[0].output_tensors[0]
        z_log_var = self._model.get_layer('z_log_var').inbound_nodes[0].output_tensors[0]

        xent_loss = self.n_inputs * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var
                                - K.square(z_mean)
                                - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = {
            'enc': self._encoder.get_weights(),
            #'dec': self._decoder.get_weights()
        }

        return params

    def _load_model(self, config_file):
        self._model = load_model(filepath=config_file, custom_objects={'_vae_loss': self._vae_loss})