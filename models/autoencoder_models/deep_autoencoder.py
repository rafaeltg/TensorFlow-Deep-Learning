from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import Input, Dense
import keras.models as kmodels

import utils.utilities as utils
from models.base.unsupervised_model import UnsupervisedModel


class DeepAutoencoder(UnsupervisedModel):

    """ Implementation of Deep Autoencoders.
    """

    def __init__(self,
                 model_name='deep_ae',
                 main_dir='deep_ae/',
                 n_hidden=list([256, 128, 64]),
                 enc_act_func='relu',
                 dec_act_func='linear',
                 loss_func='mean_squared_error',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 verbose=0,
                 seed=-1):

        """
        :param n_hidden: Number of hidden units of each layer
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid', 'relu', 'linear']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'relu', 'linear']
        :param loss_func: Cost function. ['mean_squared_error', 'cross_entropy', 'softmax_cross_entropy']
        :param num_epochs: Number of epochs for training
        :param batch_size: Size of each mini-batch
        :param opt: Which tensorflow optimizer to use. ['gradient_descent', 'momentum', 'ada_grad', 'adam', 'rms_prop']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
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
        assert len(n_hidden) > 0
        assert all([l > 0 for l in n_hidden])
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions

        self.n_hidden = n_hidden
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_inputs):

        """ Create the encoding and the decoding layers of the deep autoencoder.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.model_name))

        self._encode_layer = self._input
        for l in self.n_hidden:
            self._encode_layer = Dense(output_dim=l,
                                       activation=self.enc_act_func)(self._encode_layer)

        self._decode_layer = self._encode_layer
        for l in self.n_hidden[1::-1] + [n_inputs]:
            self._decode_layer = Dense(output_dim=l,
                                       activation=self.dec_act_func)(self._decode_layer)

    def _create_encoder_model(self):

        """ Create the encoding layer of the deep autoencoder.
        :return: self
        """

        self.logger.info('Creating {} encoder model'.format(self.model_name))

        # This model maps an input to its encoded representation
        self._encoder = kmodels.Model(input=self._input, output=self._encode_layer)

        self.logger.info('Done creating {} encoder model'.format(self.model_name))

    def _create_decoder_model(self):

        """ Create the decoding layers of the deep autoencoder.
        :return: self
        """

        self.logger.info('Creating {} decoder model'.format(self.model_name))

        self._encoded_input = Input(shape=(self.n_hidden[-1],))

        decoder_layer = self._encoded_input
        for l in self._model.layers[len(self.n_hidden)+1:]:
            decoder_layer = l(decoder_layer)

        # create the decoder model
        self._decoder = kmodels.Model(input=self._encoded_input, output=decoder_layer)

        self.logger.info('Done creating {} decoding layer'.format(self.model_name))

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = {
            'enc': self._encoder.get_weights(),
            'dec': self._decoder.get_weights()
        }

        return params
