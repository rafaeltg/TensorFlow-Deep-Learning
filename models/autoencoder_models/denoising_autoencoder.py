from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from models.autoencoder_models.autoencoder import Autoencoder


class DenoisingAutoencoder(Autoencoder):

    """ Implementation of a Denoising Autoencoder.
    """

    def __init__(self,
                 model_name='dae',
                 main_dir='dae/',
                 n_hidden=256,
                 enc_act_func='relu',
                 dec_act_func='linear',
                 loss_func='mean_squared_error',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 corr_type='gaussian',
                 corr_scale=0.1,
                 corr_keep_prob=0.9,
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
        :param corr_scale:
        :param corr_keep_prob:
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
        assert corr_scale > 0 if corr_type == 'gaussian' else True
        assert 0 <= corr_keep_prob <= 1.0 if corr_type == 'masking' else True

        self.corr_type = corr_type
        self.corr_scale = corr_scale
        self.corr_keep_prob = corr_keep_prob

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def fit(self, x_train, x_valid=None):

        """
        :param x_train: Training data. shape(n_samples, n_features)
        :param x_valid: Validation data. shape(n_samples, n_features)
        :return:
        """

        self.logger.info('Starting {} unsupervised training...'.format(self.model_name))

        self.build_model(x_train.shape[1])

        corr_x_train = self._corrupt_input(x_train)
        corr_x_valid = self._corrupt_input(x_valid)

        self._model.fit(x=corr_x_train,
                        y=x_train,
                        nb_epoch=self.num_epochs,
                        batch_size=self.batch_size,
                        shuffle=True,
                        validation_data=(corr_x_valid, x_valid),
                        verbose=self.verbose)

        self.logger.info('Done {} unsupervised training...'.format(self.model_name))

    def _corrupt_input(self, x):

        """
        :param x:
        :return:
        """

        self.logger.info('Corrupting Input Data')

        corr_x = None

        if self.corr_type == 'masking':
            corr_x = x * (np.random.uniform(0, 1, x.shape) < self.corr_keep_prob).astype(int)

        elif self.corr_type == 'gaussian':
            corr_x = x + self.corr_scale * np.random.normal(loc=0.0, scale=1.0, size=x.shape)

        return corr_x
