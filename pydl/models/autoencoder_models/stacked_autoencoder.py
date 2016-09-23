from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Dropout

import pydl.utils.utilities as utils
from pydl.models.autoencoder_models.autoencoder import Autoencoder
from pydl.models.base.supervised_model import SupervisedModel


class StackedAutoencoder(SupervisedModel):

    """ Implementation of Stacked Autoencoders.
    """

    def __init__(self,
                 model_name='sae',
                 main_dir='sae/',
                 layers=list([128, 64, 32]),
                 enc_act_func=list(['relu']),
                 dec_act_func=list(['linear']),
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
        :param loss_func: Loss function of each autoencoder.
        :param num_epochs: Number of epochs for training of each autoencoder.
        :param batch_size: Size of mini-batch of each autoencoder.
        :param opt: Optimization function of each autoencoder.
        :param learning_rate: Initial learning rate of each autoencoder.
        :param momentum: Momentum parameter of each autoencoder.
        :param hidden_dropout: Hidden layers dropout parameter for the finetuning step.
        :param finetune_loss_func: loss function for the finetuning step.
        :param finetune_enc_act_func: Finetuning step  hidden layers activation function.
        :param finetune_dec_act_func: Finetuning step output layer activation function.
        :param finetune_opt: Optimization function for the finetuning step.
        :param finetune_learning_rate: Learning rate for the finetuning.
        :param finetune_momentum: Momentum for the finetuning.
        :param finetune_num_epochs: Number of epochs for the finetuning.
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose:
        """

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         loss_func=finetune_loss_func,
                         num_epochs=finetune_num_epochs,
                         batch_size=finetune_batch_size,
                         opt=finetune_opt,
                         learning_rate=finetune_learning_rate,
                         momentum=finetune_momentum,
                         verbose=verbose,
                         seed=seed)

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert len(layers) > 0
        assert all([l > 0 for l in layers])
        assert finetune_enc_act_func in utils.valid_act_functions
        assert finetune_dec_act_func in utils.valid_act_functions
        assert 0 <= hidden_dropout <= 1.0

        # Autoencoder parameters
        ae_args = {
            'layers':        layers,
            'enc_act_func':  enc_act_func,
            'dec_act_func':  dec_act_func,
            'loss_func':     loss_func,
            'num_epochs':    num_epochs,
            'batch_size':    batch_size,
            'opt':           opt,
            'learning_rate': learning_rate,
            'momentum':      momentum,
        }

        self.ae_args = utils.expand_args(ae_args)

        # Finetuning parameters
        self.pretrain_params = []
        self.enc_act_func = finetune_enc_act_func
        self.dec_act_func = finetune_dec_act_func
        self.dropout = hidden_dropout

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
                                  activation=self.enc_act_func))

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

        self.logger.info('Starting {} unsupervised pretraining...'.format(self.model_name))

        next_train = x_train
        next_valid = x_valid

        for l in range(len(self.ae_args['layers'])):

            self.logger.info('Pre-training layer {}'.format(l))

            ae = self._get_autoencoder(l)

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

        self.logger.info('Done {} unsupervised pretraining...'.format(self.model_name))

    def _get_autoencoder(self, n):

        return Autoencoder(model_name='{}_ae_{}'.format(self.model_name, n),
                           main_dir=self.main_dir,
                           n_hidden=self.ae_args['layers'][n],
                           enc_act_func=self.ae_args['enc_act_func'][n],
                           dec_act_func=self.ae_args['dec_act_func'][n],
                           loss_func=self.ae_args['loss_func'][n],
                           num_epochs=self.ae_args['num_epochs'][n],
                           batch_size=self.ae_args['batch_size'][n],
                           opt=self.ae_args['opt'][n],
                           learning_rate=self.ae_args['learning_rate'][n],
                           momentum=self.ae_args['momentum'][n],
                           verbose=self.verbose)

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
