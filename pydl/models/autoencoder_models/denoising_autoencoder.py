from keras.layers.noise import GaussianDropout, GaussianNoise

from .autoencoder import Autoencoder


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
                 corr_type='gaussian',
                 corr_param=0.1,
                 **kwargs):

        """
        :param n_hidden: number of hidden units
        :param enc_act_func: Activation function for the encoder.
        :param dec_act_func: Activation function for the decoder.
        :param l1_reg: L1 weight regularization penalty, also known as LASSO.
        :param l2_reg: L2 weight regularization penalty, also known as weight decay, or Ridge.
        :param corr_type: Type of input corruption. ["masking", "gaussian"]
        :param corr_param: 'scale' parameter for Additive Gaussian Corruption ('gaussian') or
                           'noise_level' - fraction of the entries that will be set to 0 (Masking Corruption - 'masking')
        """

        self.corr_type = corr_type
        self.corr_param = corr_param

        super().__init__(name=name,
                         n_hidden=n_hidden,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         **kwargs)

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
