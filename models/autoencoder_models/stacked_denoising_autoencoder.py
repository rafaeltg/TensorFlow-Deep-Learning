from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils.utilities as utils
from models.autoencoder_models.denoising_autoencoder import DenoisingAutoencoder
from models.autoencoder_models.stacked_autoencoder import StackedAutoencoder


class StackedDenoisingAutoencoder(StackedAutoencoder):

    """ Implementation of Stacked Denoising Autoencoders using TensorFlow.
    """

    def __init__(self,
                 model_name='sdae',
                 main_dir='sdae/',
                 layers=list([256, 128]),
                 enc_act_func=list(['tanh']),
                 dec_act_func=list(['none']),
                 cost_func=list(['rmse']),
                 num_epochs=list([10]),
                 batch_size=list([10]),
                 opt=list(['adam']),
                 learning_rate=list([0.01]),
                 momentum=list([0.5]),
                 corr_type=list(['masking']),
                 corr_scale=list([0.1]),
                 corr_keep_prob=list([0.9]),
                 rho=list([0.001]),
                 n_beta=list([3.0]),
                 n_lambda=list([0.0001]),
                 hidden_dropout=1.0,
                 finetune_cost_func='rmse',
                 finetune_act_func='relu',
                 finetune_opt='adam',
                 finetune_learning_rate=0.001,
                 finetune_momentum=0.5,
                 finetune_num_epochs=10,
                 finetune_batch_size=100,
                 seed=-1,
                 verbose=0,
                 task='regression'):

        """
        :param layers: list containing the hidden units for each layer
        :param enc_act_func: Activation function for the encoder. ['sigmoid', 'tanh', 'relu', 'none']
        :param dec_act_func: Activation function for the decoder. ['sigmoid', 'tanh', 'none']
        :param cost_func: Cost function. ['cross_entropy', 'rmse', 'softmax_cross_entropy', 'sparse'].
        :param num_epochs: Number of epochs for training.
        :param batch_size: Size of each mini-batch.
        :param opt: Optimizer to use. string, default 'gradient_descent'. ['gradient_descent', 'ada_grad', 'momentum', 'rms_prop']
        :param learning_rate: Initial learning rate.
        :param momentum: 'Momentum parameter.
        :param rho:
        :param n_beta:
        :param n_lambda:
        :param corr_type: type of input corruption. ["masking", "gaussian"]
        :param corr_scale: scale parameter for Aditive Gaussian Corruption ('gaussian')
        :param corr_keep_prob: keep_prob parameter for Masking Corruption ('masking')
        :param hidden_dropout: hidden layers dropout parameter.
        :param finetune_cost_func: Cost function for the fine tunning step. ['cross_entropy', 'rmse', 'softmax_cross_entropy', 'sparse']
        :param finetune_act_func: activation function for the finetuning step. ['sigmoid', 'tanh', 'relu', 'none']
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_learning_rate: learning rate for the finetuning.
        :param finetune_momentum: momentum for the finetuning.
        :param finetune_num_epochs: Number of epochs for the finetuning.
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param task: ['regression', 'classification']
        """

        print('{} __init__'.format(__class__.__name__))

        super().__init__(model_name,
                         main_dir,
                         layers,
                         enc_act_func,
                         dec_act_func,
                         cost_func,
                         num_epochs,
                         batch_size,
                         opt,
                         learning_rate,
                         momentum,
                         rho,
                         n_beta,
                         n_lambda,
                         hidden_dropout,
                         finetune_cost_func,
                         finetune_act_func,
                         finetune_opt,
                         finetune_learning_rate,
                         finetune_momentum,
                         finetune_num_epochs,
                         finetune_batch_size,
                         seed,
                         verbose,
                         task)

        # Denoising Autoencoder parameters
        self.ae_args['corr_type']      = corr_type
        self.ae_args['corr_scale']     = corr_scale
        self.ae_args['corr_keep_prob'] = corr_keep_prob

        self.ae_args = utils.expand_args(self.ae_args)

        print('Done {} __init__'.format(__class__.__name__))

    def _create_autoencoders(self):

        """  Create Denoising Autoencoder Objects
        :return: self
        """

        print('Creating {} pretrain nodes...'.format(self.model_name))

        self.autoencoders = []
        self.autoencoder_graphs = []

        for l, layer in enumerate(self.layers):

            print('l = {}, layer = {}'.format(l, layer))

            self.autoencoders.append(DenoisingAutoencoder(model_name='{}_dae_{}'.format(self.model_name, l),
                                                          main_dir=self.main_dir,
                                                          n_hidden=layer,
                                                          enc_act_func=self.ae_args['enc_act_func'][l],
                                                          dec_act_func=self.ae_args['dec_act_func'][l],
                                                          cost_func=self.ae_args['cost_func'][l],
                                                          num_epochs=self.ae_args['num_epochs'][l],
                                                          batch_size=self.ae_args['batch_size'][l],
                                                          opt=self.ae_args['opt'][l],
                                                          learning_rate=self.ae_args['learning_rate'][l],
                                                          momentum=self.ae_args['momentum'][l],
                                                          corr_type=self.ae_args['corr_type'][l],
                                                          corr_scale=self.ae_args['corr_scale'][l],
                                                          corr_keep_prob=self.ae_args['corr_keep_prob'][l],
                                                          rho=self.ae_args['rho'][l],
                                                          n_beta=self.ae_args['n_beta'][l],
                                                          n_lambda=self.ae_args['n_lambda'][l],
                                                          verbose=self.verbose))

            self.autoencoder_graphs.append(tf.Graph())

        print('Done creating {} pretrain nodes...'.format(self.model_name))
