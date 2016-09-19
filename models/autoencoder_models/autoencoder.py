from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.models as kmodels
from keras.layers import Input, Dense

import utils.utilities as utils
from models.base.unsupervised_model import UnsupervisedModel


class Autoencoder(UnsupervisedModel):

    """ Implementation of a Autoencoder.
    """

    def __init__(self,
                 model_name='ae',
                 main_dir='ae/',
                 n_hidden=32,
                 enc_act_func='relu',
                 dec_act_func='linear',
                 loss_func='mse',
                 num_epochs=10,
                 batch_size=100,
                 opt='rmsprop',
                 learning_rate=0.01,
                 momentum=0.5,
                 verbose=0,
                 seed=42):

        """
        :param n_hidden: number of hidden units
        :param enc_act_func: Activation function for the encoder.
        :param dec_act_func: Activation function for the decoder.
        :param loss_func: Loss function.
        :param num_epochs: Number of epochs for training.
        :param batch_size: Size of each mini-batch.
        :param opt: Which optimizer to use.
        :param learning_rate: Initial learning rate.
        :param momentum: Momentum parameter.
        :param verbose: Level of verbosity. 0 - silent, 1 - print
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

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

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert n_hidden > 0
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions

        self.n_hidden = n_hidden
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_inputs):

        """ Create the encoding and the decoding layers of the autoencoder.
        :param n_inputs: Input size
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.model_name))

        self._encode_layer = Dense(output_dim=self.n_hidden,
                                   activation=self.enc_act_func)(self._input)

        self._decode_layer = Dense(output_dim=n_inputs,
                                   activation=self.dec_act_func)(self._encode_layer)

    def _create_encoder_model(self):

        """ Create the model that maps an input to its encoded representation.
        :return: self
        """

        self.logger.info('Creating {} encoder model'.format(self.model_name))

        self._encoder = kmodels.Model(input=self._input, output=self._encode_layer)

        self.logger.info('Done creating {} encoder model'.format(self.model_name))

    def _create_decoder_model(self):

        """ Create the model that maps an encoded input to the original values
        :return: self
        """

        self.logger.info('Creating {} decoder model'.format(self.model_name))

        self._encoded_input = Input(shape=(self.n_hidden,))

        # retrieve the last layer of the autoencoder model
        decoder_layer = self._model.layers[-1]

        self._decoder = kmodels.Model(input=self._encoded_input, output=decoder_layer(self._encoded_input))

        self.logger.info('Done creating {} decoding layer'.format(self.model_name))

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = {
            'enc': self._encoder.get_weights()
            #'dec': self._decoder.get_weights()
        }

        return params
