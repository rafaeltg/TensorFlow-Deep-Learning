from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.models as kmodels
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import objectives

import utils.utilities as utils
from models.base.unsupervised_model import UnsupervisedModel


class VariationalAutoencoder(UnsupervisedModel):

    """ Implementation of a Variational Autoencoder.
    """

    def __init__(self,
                 model_name='vae',
                 main_dir='vae/',
                 n_latent=10,
                 n_hidden=list([256, 128]),
                 enc_act_func='relu',
                 dec_act_func='linear',
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
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid', 'relu', 'linear']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'relu', 'linear']
        :param num_epochs: Number of epochs for training
        :param batch_size: Size of each mini-batch
        :param opt: Which tensorflow optimizer to use. ['sgd', 'momentum', 'ada_grad', 'adam', 'rmsprop']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         loss_func='custom',
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert n_latent > 0
        assert all([l > 0 for l in n_hidden])
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions

        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func

        self.z_mean_layer = None
        self.z_log_var_layer = None

        self.loss_func = self._vae_loss
        self.n_input = None

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_inputs):

        """ Create the encoding and the decoding layers of the variational autoencoder.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.model_name))

        # Encode layers
        self._encode_layer = self._input
        for l in self.n_hidden:
            self._encode_layer = Dense(output_dim=l,
                                       activation=self.enc_act_func)(self._encode_layer)

        self.z_mean_layer = Dense(output_dim=self.n_latent)(self._encode_layer)
        self.z_log_var_layer = Dense(output_dim=self.n_latent)(self._encode_layer)

        z = Lambda(self.sampling, output_shape=(self.n_latent,))([self.z_mean_layer, self.z_log_var_layer])

        # Decode layers
        self._decode_layer = z
        for l in self.n_hidden[::-1]:
            self._decode_layer = Dense(output_dim=l,
                                       activation=self.dec_act_func)(self._decode_layer)

        self._decode_layer = Dense(n_inputs, activation='sigmoid')(self._decode_layer)

    def _create_encoder_model(self):

        """ Create the encoding layer of the autoencoder.
        :return: self
        """

        self.logger.info('Creating {} encoder model'.format(self.model_name))

        # This model maps an input to its encoded representation
        self._encoder = kmodels.Model(input=self._input, output=self.z_mean_layer)

        self.logger.info('Done creating {} encoder model'.format(self.model_name))

    def _create_decoder_model(self):

        """ Create the decoding layers of the variational autoencoder.
        :return: self
        """

        self._model.summary()

        self.logger.info('Creating {} decoder model'.format(self.model_name))

        self._encoded_input = Input(shape=(self.n_latent,))

        decoder_layer = self._encoded_input
        print(self._model.layers[len(self.n_hidden)+4:])

        for l in self._model.layers[len(self.n_hidden)+3:]:
            decoder_layer = l(decoder_layer)

        # create the decoder model
        self._decoder = kmodels.Model(input=self._encoded_input, output=decoder_layer)

        self.logger.info('Done creating {} decoding layer'.format(self.model_name))

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.n_latent), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        xent_loss = self.n_input * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var_layer
                                - K.square(self.z_mean_layer)
                                - K.exp(self.z_log_var_layer), axis=-1)
        return xent_loss + kl_loss

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = {
            'enc': self._encoder.layers[1].get_weights(),
            'dec': self._decoder.layers[1].get_weigths()
        }

        return params
