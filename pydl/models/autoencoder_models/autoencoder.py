from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
import keras.models as kmodels
from keras.layers import Input, Dense
from keras.regularizers import l1l2

from pydl.utils.utilities import valid_act_functions
from ..base import UnsupervisedModel


class Autoencoder(UnsupervisedModel):

    """ Implementation of an Autoencoder.
    """

    def __init__(self,
                 name='ae',
                 n_hidden=32,
                 enc_act_func='relu',
                 dec_act_func='linear',
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

        self.n_hidden = n_hidden
        self.enc_act_func = enc_act_func
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

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert self.n_hidden > 0
        assert self.enc_act_func in valid_act_functions
        assert self.dec_act_func in valid_act_functions

    def _create_layers(self, input_layer):

        """ Create the encoding and the decoding layers of the autoencoder.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.name))

        encode_layer = Dense(name='encoder',
                             output_dim=self.n_hidden,
                             activation=self.enc_act_func,
                             W_regularizer=l1l2(self.l1_reg, self.l2_reg),
                             b_regularizer=l1l2(self.l1_reg, self.l2_reg))(input_layer)

        n_inputs = K.int_shape(input_layer)[-1]
        self._decode_layer = Dense(name='decoder',
                                   output_dim=n_inputs,
                                   activation=self.dec_act_func)(encode_layer)

    def _create_encoder_model(self):

        """ Create the model that maps an input to its encoded representation.
        :return: self
        """

        self.logger.info('Creating {} encoder model'.format(self.name))

        self._encoder = kmodels.Model(input=self._model.layers[0].inbound_nodes[0].output_tensors,
                                      output=self._model.get_layer('encoder').inbound_nodes[0].output_tensors)

        self.logger.info('Done creating {} encoder model'.format(self.name))

    def _create_decoder_model(self):

        """ Create the model that maps an encoded input to the original values
        :return: self
        """

        self.logger.info('Creating {} decoder model'.format(self.name))

        encoded_input = Input(shape=(self.n_hidden,))

        # retrieve the last layer of the autoencoder model
        decoder_layer = self._model.get_layer('decoder')

        self._decoder = kmodels.Model(input=encoded_input,
                                      output=decoder_layer(encoded_input))

        self.logger.info('Done creating {} decoding layer'.format(self.name))

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = {
            'enc': self._encoder.get_weights()
            #'dec': self._decoder.get_weights()
        }

        return params
