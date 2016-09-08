from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils.utilities as utils
from models.base.supervised_model import SupervisedModel
from models.autoencoder_models.autoencoder import Autoencoder
from keras.layers import Dense, Dropout


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

        # Autoencoders list
        self.autoencoders = []

        # Finetuning parameters
        self.enc_act_func = finetune_enc_act_func
        self.dec_act_func = finetune_dec_act_func
        self.dropout = hidden_dropout

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_autoencoders(self):

        """ Create Autoencoder Objects used for unsupervised pretraining
        :return: self
        """

        self.logger.info('Creating {} pretrain nodes...'.format(self.model_name))

        for l, layer in enumerate(self.ae_args['layers']):

            self.logger.info('l = {}, layer = {}'.format(l, layer))

            self.autoencoders.append(Autoencoder(model_name='{}_ae_{}'.format(self.model_name, l),
                                                 main_dir=self.main_dir,
                                                 n_hidden=layer,
                                                 enc_act_func=self.ae_args['enc_act_func'][l],
                                                 dec_act_func=self.ae_args['dec_act_func'][l],
                                                 loss_func=self.ae_args['loss_func'][l],
                                                 num_epochs=self.ae_args['num_epochs'][l],
                                                 batch_size=self.ae_args['batch_size'][l],
                                                 opt=self.ae_args['opt'][l],
                                                 learning_rate=self.ae_args['learning_rate'][l],
                                                 momentum=self.ae_args['momentum'][l],
                                                 verbose=self.verbose))

        self.logger.info('Done creating {} pretrain nodes...'.format(self.model_name))

    def _create_layers(self, n_input, n_output):

        """ Create the finetuning model
        :param n_input:
        :param n_output:
        :return: self
        """

        # Hidden layers
        for n, l in enumerate(self.ae_args['layers']):

            if self.dropout < 1:
                self._model_layers = Dropout(p=self.dropout)(self._model_layers)

            # Get autoencoder parameters
            # params[0] = weights
            # params[1] = biases
            params = self.autoencoders[n].get_model_parameters()['enc']

            self._model_layers = Dense(output_dim=l,
                                       weights=params,
                                       activation=self.enc_act_func)(self._model_layers)

        # Output layer
        if self.dropout < 1:
            self._model_layers = Dropout(p=self.dropout)(self._model_layers)

        self._model_layers = Dense(output_dim=n_output,
                                   init='glorot_normal',
                                   activation=self.dec_act_func)(self._model_layers)

    def _pretrain(self, x_train, x_valid):

        """ Perform unsupervised pretraining of the stack of autoencoders.
        :param x_train: training set
        :param x_valid: validation set
        :return: self
        """

        self.logger.info('Starting {} unsupervised pretraining...'.format(self.model_name))

        self._create_autoencoders()

        next_train = x_train
        next_valid = x_valid

        for l, autoenc in enumerate(self.autoencoders):

            self.logger.info('Pre-training layer {}'.format(l+1))

            # Pretrain a single autoencoder
            autoenc.fit(next_train, next_valid)

            # Encode the data for the next layer.
            next_train = autoenc.transform(data=next_train)
            next_valid = autoenc.transform(data=next_valid)

        self.logger.info('Done {} unsupervised pretraining...'.format(self.model_name))

    def fit(self, x_train, y_train, x_valid, y_valid):

        """
        :param x_train:
        :param y_train:
        :param x_valid:
        :param y_valid:
        :return: self
        """

        self._pretrain(x_train, x_valid)

        super().fit(x_train, y_train, x_valid, y_valid)
