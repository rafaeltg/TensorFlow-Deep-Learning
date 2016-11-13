from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers.noise import GaussianDropout, GaussianNoise
from pydl.models.autoencoder_models.autoencoder import Autoencoder


class DenoisingAutoencoder(Autoencoder):

    """ Implementation of a Denoising Autoencoder.
    """

    def __init__(self,
                 name='dae',
                 n_hidden=32,
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
                 corr_type='gaussian',
                 corr_param=0.1,
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
        :param corr_type: Type of input corruption. ["masking", "gaussian"]
        :param corr_param: 'scale' parameter for Aditive Gaussian Corruption ('gaussian') or
                           'keep_prob' parameter for Masking Corruption ('masking')
        :param verbose: Level of verbosity. 0 - silent, 1 - print
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

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

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert corr_type in ['masking', 'gaussian']
        assert corr_param > 0 if corr_type == 'gaussian' else True
        assert 0 <= corr_param <= 1.0 if corr_type == 'masking' else True

        self.corr_type = corr_type
        self.corr_param = corr_param

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_inputs):

        """ Create the encoding and the decoding layers of the autoencoder.
        :param n_inputs: Input size.
        :return:
        """

        self._corrupt_input()

        super()._create_layers(n_inputs)

    def _corrupt_input(self):

        """ Apply some noise to the input data.
        :return:
        """

        self.logger.info('Corrupting Input - {}'.format(self.corr_type))

        if self.corr_type == 'masking':
            self._encode_layer = GaussianDropout(p=self.corr_param)(self._input)

        else:
            self._encode_layer = GaussianNoise(sigma=self.corr_param)(self._input)
