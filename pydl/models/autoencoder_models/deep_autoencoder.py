from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.models as kmodels
import pydl.utils.utilities as utils
from keras.layers import Input, Dense
from keras.regularizers import l1l2
from pydl.models.base.unsupervised_model import UnsupervisedModel


class DeepAutoencoder(UnsupervisedModel):

    """ Implementation of Deep Autoencoders.
    """

    def __init__(self,
                 model_name='deep_ae',
                 main_dir='deep_ae/',
                 n_hidden=list([64, 32, 16]),
                 enc_act_func='relu',
                 dec_act_func='linear',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 loss_func='mse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 verbose=0,
                 seed=42):

        """
        :param n_hidden: Number of hidden units of each layer
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
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
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
        :param n_inputs: Input size.
        :return: self
        """

        self.logger.info('Creating {} layers'.format(self.model_name))

        self._encode_layer = self._input
        for l in self.n_hidden:
            self._encode_layer = Dense(output_dim=l,
                                       activation=self.enc_act_func,
                                       W_regularizer=l1l2(self.l1_reg, self.l2_reg),
                                       b_regularizer=l1l2(self.l1_reg, self.l2_reg))(self._encode_layer)

        self._decode_layer = self._encode_layer
        for l in self.n_hidden[1::-1] + [n_inputs]:
            self._decode_layer = Dense(output_dim=l,
                                       activation=self.dec_act_func)(self._decode_layer)

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

        encoded_input = Input(shape=(self.n_hidden[-1],))

        decoder_layer = encoded_input
        for l in self._model.layers[len(self.n_hidden)+1:]:
            decoder_layer = l(decoder_layer)

        self._decoder = kmodels.Model(input=encoded_input, output=decoder_layer)

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