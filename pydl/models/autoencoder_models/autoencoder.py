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
                 **kwargs):

        """
        :param n_hidden: number of hidden units
        :param enc_act_func: Activation function for the encoder.
        :param dec_act_func: Activation function for the decoder.
        :param l1_reg: L1 weight regularization penalty, also known as LASSO.
        :param l2_reg: L2 weight regularization penalty, also known as weight decay, or Ridge.
        """

        self.n_hidden = n_hidden
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        super().__init__(name=name, **kwargs)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert self.n_hidden > 0, 'Invalid number of hidden units!'
        assert self.enc_act_func in valid_act_functions, 'Invalid encoder activation function (%s)!' % self.enc_act_func
        assert self.dec_act_func in valid_act_functions, 'Invalid decoder activation function (%s)!' % self.dec_act_func
        assert self.l1_reg >= 0, 'Invalid l1_reg value. Must be a positive value!'
        assert self.l2_reg >= 0, 'Invalid l2_reg value. Must be a positive value!'

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

        self._encoder = kmodels.Model(input=self._model.layers[0].output,
                                      output=self._model.get_layer('encoder').output)

        self.logger.info('Done creating {} encoder model'.format(self.name))

    def _create_decoder_model(self):

        """ Create the model that maps an encoded input to the original values
        :return: self
        """

        self.logger.info('Creating {} decoder model'.format(self.name))

        encoded_input = Input(shape=(self.n_hidden,))

        self._decoder = kmodels.Model(input=encoded_input,
                                      output=self._model.get_layer('decoder')(encoded_input))

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
