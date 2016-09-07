from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.autoencoder_models.autoencoder import Autoencoder
from keras.layers.core import Dense
from keras.layers.noise import GaussianDropout, GaussianNoise

class DenoisingAutoencoder(Autoencoder):

    """ Implementation of a Denoising Autoencoder.
    """

    def __init__(self,
                 model_name='dae',
                 main_dir='dae/',
                 n_hidden=32,
                 enc_act_func='relu',
                 dec_act_func='linear',
                 loss_func='mean_squared_error',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 corr_type='gaussian',
                 corr_param=0.1,
                 verbose=0,
                 seed=-1):

        """
        :param n_hidden: number of hidden units
        :param enc_act_func: Activation function for the encoder. ['tanh', 'sigmoid', 'relu', 'linear']
        :param dec_act_func: Activation function for the decoder. ['tanh', 'sigmoid', 'relu', 'linear']
        :param loss_func: Cost function. ['mean_squared_error', 'cross_entropy', 'softmax_cross_entropy', 'sparse']
        :param num_epochs: Number of epochs
        :param batch_size: Size of each mini-batch
        :param opt: Which TensorFlow optimizer to use. ['sgd', 'momentum', 'ada_grad', 'adam', 'rms_prop']
        :param learning_rate: Initial learning rate
        :param momentum: Momentum parameter
        :param corr_type: Type of input corruption. ["masking", "gaussian"]
        :param corr_param: 'scale' parameter for Aditive Gaussian Corruption ('gaussian') or
                           'keep_prob' parameter for Masking Corruption ('masking')
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         n_hidden=n_hidden,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
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

        """
        :param n_inputs:
        :return:
        """

        self.logger.info('Creating {} layers'.format(self.model_name))

        self._corrupt_input()

        self._encode_layer = Dense(output_dim=self.n_hidden,
                                   activation=self.enc_act_func)(self._encode_layer)

        self._decode_layer = Dense(output_dim=n_inputs,
                                   activation=self.dec_act_func)(self._encode_layer)

    def _corrupt_input(self):

        """
        :return:
        """

        self.logger.info('Corrupting Input Data')

        if self.corr_type == 'masking':
            self._encode_layer = GaussianDropout(p=self.corr_param)(self._input)

        else:
            self._encode_layer = GaussianNoise(sigma=self.corr_param)(self._input)
