from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pydl.utils.utilities as utils
from keras.layers import Dense, Dropout
from keras.regularizers import l1l2
from pydl.models.autoencoder_models.autoencoder import Autoencoder
from pydl.models.base.supervised_model import SupervisedModel


class StackedAutoencoder(SupervisedModel):

    """ Implementation of Stacked Autoencoders.
    """

    def __init__(self,
                 name='sae',
                 layers=list([64, 32]),
                 enc_act_func=list(['relu']),
                 dec_act_func=list(['linear']),
                 l1_reg=list([0.0]),
                 l2_reg=list([0.0]),
                 loss_func=list(['mse']),
                 num_epochs=list([10]),
                 batch_size=list([100]),
                 opt=list(['adam']),
                 learning_rate=list([0.001]),
                 momentum=list([0.5]),
                 hidden_dropout=1.0,
                 finetune_loss_func='mse',
                 finetune_enc_act_func='linear',
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
        :param layers: List containing the hidden units for each layer.
        :param enc_act_func: Activation function of each autoencoder.
        :param dec_act_func: Activation function of each autoencoder.
        :param l1_reg: L1 weight regularization penalty of each autoencoder.
        :param l2_reg: L2 weight regularization penalty of each autoencoder.
        :param loss_func: Loss function of each autoencoder.
        :param num_epochs: Number of epochs for training of each autoencoder.
        :param batch_size: Size of mini-batch of each autoencoder.
        :param opt: Optimization function of each autoencoder.
        :param learning_rate: Initial learning rate of each autoencoder.
        :param momentum: Momentum parameter of each autoencoder.
        :param hidden_dropout: Fraction of the finetuning hidden layers units to drop.
        :param finetune_loss_func: loss function for the finetuning step.
        :param finetune_enc_act_func: Finetuning step  hidden layers activation function.
        :param finetune_dec_act_func: Finetuning step output layer activation function.
        :param finetune_l1_reg:
        :param finetune_l2_reg:
        :param finetune_opt: Optimization function for the finetuning step.
        :param finetune_learning_rate: Learning rate for the finetuning.
        :param finetune_momentum: Momentum for the finetuning.
        :param finetune_num_epochs: Number of epochs for the finetuning.
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose:
        """

        super().__init__(name=name,
                         layers=layers,
                         enc_act_func=finetune_enc_act_func,
                         dec_act_func=finetune_dec_act_func,
                         loss_func=finetune_loss_func,
                         l1_reg=finetune_l1_reg,
                         l2_reg=finetune_l2_reg,
                         dropout=hidden_dropout,
                         num_epochs=finetune_num_epochs,
                         batch_size=finetune_batch_size,
                         opt=finetune_opt,
                         learning_rate=finetune_learning_rate,
                         momentum=finetune_momentum,
                         seed=seed,
                         verbose=verbose)

        # Autoencoder parameters
        self.ae_type = Autoencoder
        self.ae_enc_act_func = enc_act_func
        self.ae_dec_act_func = dec_act_func
        self.ae_l1_reg = l1_reg
        self.ae_l2_reg = l2_reg
        self.ae_loss_func = loss_func
        self.ae_num_epochs = num_epochs
        self.ae_batch_size = batch_size
        self.ae_opt = opt
        self.ae_learning_rate = learning_rate
        self.ae_momentum = momentum

        # Finetuning parameters
        self.pretrain_params = []

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, input_shape, n_output):

        """ Create the finetuning model
        :param input_shape:
        :param n_output:
        :return: self
        """

        # Hidden layers
        for n, l in enumerate(self.ae_args['layers']):

            self._model.add(Dropout(p=self.dropout,
                                    input_shape=[input_shape[1] if n == 0 else None]))

            self._model.add(Dense(output_dim=l,
                                  weights=self.pretrain_params[n],
                                  activation=self.enc_act_func,
                                  W_regularizer=l1l2(self.l1_reg, self.l2_reg),
                                  b_regularizer=l1l2(self.l1_reg, self.l2_reg)))

        # Output layer
        self._model.add(Dropout(p=self.dropout))
        self._model.add(Dense(output_dim=n_output,
                              activation=self.dec_act_func))

    def _pretrain(self, x_train, x_valid=None):

        """ Perform unsupervised pretraining of the stack of autoencoders.
        :param x_train: training set
        :param x_valid: validation set
        :return: self
        """

        self.logger.info('Starting {} unsupervised pretraining...'.format(self.name))

        self.pretrain_params = []

        next_train = x_train
        next_valid = x_valid

        aes = self._get_autoencoders_params()

        for i, ae_params in enumerate(aes):

            self.logger.info('Pre-training layer {}'.format(i))

            ae = self.ae_type(**ae_params)

            # Pretrain a single autoencoder
            ae.fit(next_train, next_valid)

            # Get autoencoder parameters
            # params[0] = weights
            # params[1] = biases
            self.pretrain_params.append(ae.get_model_parameters()['enc'])

            # Encode the data for the next layer
            next_train = ae.transform(data=next_train)

            if x_valid:
                next_valid = ae.transform(data=next_valid)

        self.logger.info('Done {} unsupervised pretraining...'.format(self.name))

    def _get_ae_args(self):
        return {
            'enc_act_func':  self.ae_enc_act_func,
            'dec_act_func':  self.ae_dec_act_func,
            'l1_reg':        self.ae_l1_reg,
            'l2_reg':        self.ae_l2_reg,
            'loss_func':     self.ae_loss_func,
            'num_epochs':    self.ae_num_epochs,
            'batch_size':    self.ae_batch_size,
            'opt':           self.ae_opt,
            'learning_rate': self.ae_learning_rate,
            'momentum':      self.ae_momentum,
        }

    def _get_autoencoders_params(self):
        ae_args = utils.expand_args(self.layers, self._get_ae_args())
        aes_params = []

        for i, l in enumerate(self.layers):
            ae_params = {
                'name': self.name + '_ae_{}'.format(i),
                'n_hidden': l,
                'verbose': self.verbose}
            ae_params.update(dict((k, v[i]) for k, v in ae_args.items()))
            aes_params.append(ae_params)

        return aes_params

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):

        """
        :param x_train:
        :param y_train:
        :param x_valid:
        :param y_valid:
        :return: self
        """

        self._pretrain(x_train, x_valid)

        super().fit(x_train, y_train, x_valid, y_valid)
