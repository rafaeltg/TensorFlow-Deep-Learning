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
                 enc_activation='tanh',
                 dec_activation='tanh',
                 **kwargs):

        """
        :param n_hidden: number of hidden units
        :param time_steps:
        :param enc_activation: Activation function for the encoder.
        :param dec_activation: Activation function for the decoder.
        :param l1_reg: L1 weight regularization penalty, also known as LASSO.
        :param l2_reg: L2 weight regularization penalty, also known as weight decay, or Ridge.
        """

        self.time_steps = time_steps

        super().__init__(name=name,
                         n_hidden=n_hidden,
                         enc_activation=enc_activation,
                         dec_activation=dec_activation,
                         **kwargs)

    def validate_params(self):
        super().validate_params()
        assert self.time_steps > 0

    def _create_layers(self, input_layer):

        """ Create the encoding and the decoding layers of the sequence-to-sequence autoencoder.
        :return: self
        """

        encode_layer = LSTM(name='encoder',
                            units=self.n_hidden,
                            activation=self.enc_activation)(input_layer)

        n_inputs = K.int_shape(input_layer)[-1]
        decoded = RepeatVector(n=self.time_steps)(encode_layer)
        self._decode_layer = LSTM(name='decoder',
                                  units=n_inputs,
                                  activation=self.dec_activation,
                                  return_sequences=True)(decoded)

    def _create_decoder_model(self):

        """ Create the model that maps an encoded input to the original values
        :return: self
        """

        encoded_input = Input(shape=(self.n_hidden,))

        # retrieve the last layer of the autoencoder model
        decoder_layer = RepeatVector(n=self.time_steps)(encoded_input)
        decoder_layer = self._model.get_layer('decoder')(decoder_layer)

        self._decoder = kmodels.Model(inputs=encoded_input, outputs=decoder_layer)