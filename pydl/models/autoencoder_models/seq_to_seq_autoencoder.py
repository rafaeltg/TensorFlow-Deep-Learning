from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
import keras.models as kmodels
from keras.layers import Input, LSTM, RepeatVector

from .autoencoder import Autoencoder


class SeqToSeqAutoencoder(Autoencoder):

    """ Implementation of a Sequence-to-Sequence Autoencoder.
    """

    def __init__(self,
                 name='ae',
                 n_hidden=32,
                 time_steps=1,
                 enc_act_func='tanh',
                 dec_act_func='tanh',
                 l1_reg=0.0,
                 l2_reg=0.0,
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
        :param time_steps:
        :param enc_act_func: Activation function for the encoder.
        :param dec_act_func: Activation function for the decoder.
        :param l1_reg: L1 weight regularization penalty, also known as LASSO.
        :param l2_reg: L2 weight regularization penalty, also known as weight decay, or Ridge.
        :param loss_func: Loss function.
        :param num_epochs: Number of epochs for training.
        :param batch_size: Size of each mini-batch.
        :param opt: Which optimizer to use.
        :param learning_rate: Initial learning rate.
        :param momentum: Momentum parameter.
        :param verbose: Level of verbosity. 0 - silent, 1 - print
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        self.time_steps = time_steps

        super().__init__(name=name,
                         n_hidden=n_hidden,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         loss_func=loss_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         verbose=verbose,
                         seed=seed)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert self.time_steps > 0

    def _create_layers(self, input_layer):

        """ Create the encoding and the decoding layers of the sequence-to-sequence autoencoder.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.name))

        encode_layer = LSTM(name='encoder',
                            output_dim=self.n_hidden,
                            activation=self.enc_act_func)(input_layer)

        n_inputs = K.int_shape(input_layer)[-1]
        decoded = RepeatVector(n=self.time_steps)(encode_layer)
        self._decode_layer = LSTM(name='decoder',
                                  output_dim=n_inputs,
                                  activation=self.dec_act_func,
                                  return_sequences=True)(decoded)

    def _create_decoder_model(self):

        """ Create the model that maps an encoded input to the original values
        :return: self
        """

        self.logger.info('Creating {} decoder model'.format(self.name))

        encoded_input = Input(shape=(self.n_hidden,))

        # retrieve the last layer of the autoencoder model
        decoder_layer = RepeatVector(n=self.time_steps)(encoded_input)
        decoder_layer = self._model.get_layer('decoder')(decoder_layer)

        self._decoder = kmodels.Model(input=encoded_input,
                                      output=decoder_layer)

        self.logger.info('Done creating {} decoding layer'.format(self.name))
