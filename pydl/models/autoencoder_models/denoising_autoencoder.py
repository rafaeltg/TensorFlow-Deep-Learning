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
                           'noise_level' - fraction of the entries that will be set to 0 (Masking Corruption - 'masking')
        :param verbose: Level of verbosity. 0 - silent, 1 - print
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        self.corr_type = corr_type
        self.corr_param = corr_param

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
        assert self.corr_type in ['masking', 'gaussian'], 'Invalid corruption type'
        assert self.corr_param > 0 if self.corr_type == 'gaussian' else True, 'Invalid scale parameter for gaussian corruption'
        assert 0 <= self.corr_param <= 1.0 if self.corr_type == 'masking' else True, 'Invalid keep_prob parameter for masking corruption'

    def _create_layers(self, input_layer):

        # Corrupt the input
        if self.corr_type == 'masking':
            corr_input = GaussianDropout(self.corr_param)(input_layer)
        else:
            corr_input = GaussianNoise(self.corr_param)(input_layer)

        super()._create_layers(corr_input)
