import keras.backend as K
import keras.models as kmodels
from keras.layers import Input, Dense
from keras.regularizers import l1_l2
from ..base import UnsupervisedModel
from ..utils import valid_act_functions


class Autoencoder(UnsupervisedModel):

    """ Implementation of an Autoencoder.
    """

    def __init__(self,
                 name='ae',
                 n_hidden=32,
                 enc_activation='relu',
                 dec_activation='linear',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 **kwargs):

        """
        :param n_hidden: number of hidden units
        :param enc_activation: Activation function for the encoder.
        :param dec_activation: Activation function for the decoder.
        :param l1_reg: L1 weight regularization penalty, also known as LASSO.
        :param l2_reg: L2 weight regularization penalty, also known as weight decay, or Ridge.
        """

        super().__init__(name=name,
                         n_hidden=n_hidden,
                         enc_activation=enc_activation,
                         dec_activation=dec_activation,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         **kwargs)

    def validate_params(self):
        super().validate_params()
        assert self.n_hidden > 0, 'Invalid number of hidden units!'
        assert self.enc_activation in valid_act_functions, 'Invalid encoder activation function (%s)!' % self.enc_activation
        assert self.dec_activation in valid_act_functions, 'Invalid decoder activation function (%s)!' % self.dec_activation
        assert self.l1_reg >= 0, 'Invalid l1_reg value. Must be a positive value!'
        assert self.l2_reg >= 0, 'Invalid l2_reg value. Must be a positive value!'

    def _create_layers(self, input_layer):

        """ Create the encoding and the decoding layers of the autoencoder.
        :return: self
        """

        encode_layer = Dense(name='encoder',
                             units=self.n_hidden,
                             activation=self.enc_activation,
                             kernel_regularizer=l1_l2(self.l1_reg, self.l2_reg),
                             bias_regularizer=l1_l2(self.l1_reg, self.l2_reg))(input_layer)

        n_inputs = K.int_shape(input_layer)[-1]
        self._decode_layer = Dense(name='decoder',
                                   units=n_inputs,
                                   activation=self.dec_activation)(encode_layer)

    def _create_encoder_model(self):

        """ Create the model that maps an input to its encoded representation.
        :return: self
        """

        self._encoder = kmodels.Model(inputs=self._model.layers[0].output,
                                      outputs=self._model.get_layer('encoder').output)

    def _create_decoder_model(self):

        """ Create the model that maps an encoded input to the original values
        :return: self
        """

        encoded_input = Input(shape=(self.n_hidden,))

        self._decoder = kmodels.Model(inputs=encoded_input,
                                      outputs=self._model.get_layer('decoder')(encoded_input))

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = {
            'enc': self._encoder.get_weights(),
            'dec': self._decoder.get_weights()
        }

        return params