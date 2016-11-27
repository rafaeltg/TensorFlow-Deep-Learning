from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pydl.models.autoencoder_models.denoising_autoencoder import DenoisingAutoencoder
from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder


class StackedDenoisingAutoencoder(StackedAutoencoder):

    """ Implementation of Stacked Denoising Autoencoders using TensorFlow.
    """

    def __init__(self,
                 name='sdae',
                 layers=list([64, 32]),
                 enc_act_func=list(['relu']),
                 dec_act_func=list(['linear']),
                 l1_reg=list([0.0]),
                 l2_reg=list([0.0]),
                 loss_func=list(['mse']),
                 num_epochs=list([10]),
                 batch_size=list([100]),
                 opt=list(['adam']),
                 learning_rate=list([0.01]),
                 momentum=list([0.5]),
                 corr_type=list(['gaussian']),
                 corr_param=list([0.2]),
                 hidden_dropout=1.0,
                 finetune_loss_func='mse',
                 finetune_enc_act_func='relu',
                 finetune_dec_act_func='linear',
                 finetune_l1_reg=0,
                 finetune_l2_reg=0,
                 finetune_opt='adam',
                 finetune_learning_rate=0.001,
                 finetune_momentum=0.5,
                 finetune_num_epochs=10,
                 finetune_batch_size=100,
                 seed=42,
                 verbose=0):

        """
        :param layers: list containing the hidden units for each layer
        :param enc_act_func: Activation function for the encoder.
        :param dec_act_func: Activation function for the decoder.
        :param l1_reg: L1 weight regularization penalty of each denoising autoencoder.
        :param l2_reg: L2 weight regularization penalty of each denoising autoencoder.
        :param loss_func: Loss function.
        :param num_epochs: Number of epochs for training.
        :param batch_size: Size of each mini-batch.
        :param opt: Optimizer to use. string, default 'gradient_descent'.
        :param learning_rate: Initial learning rate.
        :param momentum: 'Momentum parameter.
        :param corr_type: type of input corruption. ["masking", "gaussian"]
        :param corr_param: 'scale' parameter for Aditive Gaussian Corruption ('gaussian') or
                           'keep_prob' parameter for Masking Corruption ('masking')
        :param hidden_dropout: hidden layers dropout parameter.
        :param finetune_loss_func: Cost function for the fine tunning step.
        :param finetune_enc_act_func: finetuning step hidden layers activation function.
        :param finetune_dec_act_func: finetuning step output layer activation function.
        :param finetune_l1_reg:
        :param finetune_l2_reg:
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_learning_rate: learning rate for the finetuning.
        :param finetune_momentum: momentum for the finetuning.
        :param finetune_num_epochs: Number of epochs for the finetuning.
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(name=name,
                         layers=layers,
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
                         hidden_dropout=hidden_dropout,
                         finetune_loss_func=finetune_loss_func,
                         finetune_enc_act_func=finetune_enc_act_func,
                         finetune_dec_act_func=finetune_dec_act_func,
                         finetune_l1_reg=finetune_l1_reg,
                         finetune_l2_reg=finetune_l2_reg,
                         finetune_opt=finetune_opt,
                         finetune_learning_rate=finetune_learning_rate,
                         finetune_momentum=finetune_momentum,
                         finetune_num_epochs=finetune_num_epochs,
                         finetune_batch_size=finetune_batch_size,
                         seed=seed,
                         verbose=verbose)

        # Denoising Autoencoder parameters
        self.ae_type = DenoisingAutoencoder
        self.corr_type = corr_type
        self.corr_param = corr_param

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _get_ae_args(self):
        ae_args = super()._get_ae_args()
        ae_args['corr_type'] = self.corr_type
        ae_args['corr_param'] = self.corr_param
        return ae_args
