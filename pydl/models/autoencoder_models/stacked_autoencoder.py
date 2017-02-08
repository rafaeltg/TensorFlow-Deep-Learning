from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense
from keras.regularizers import l1l2

from .autoencoder import Autoencoder
from ..base import SupervisedModel


class StackedAutoencoder(SupervisedModel):

    """ Implementation of Stacked Autoencoders.
    """

    def __init__(self,
                 name='sae',
                 layers=None,
                 dec_act_func='linear',
                 loss_func='mse',
                 l1_reg=0,
                 l2_reg=0,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 num_epochs=100,
                 batch_size=100,
                 seed=42,
                 verbose=0):

        """
        :param layers: List containing the hidden units for each layer.
        :param loss_func: loss function for the finetuning step.
        :param dec_act_func: Finetuning step output layer activation function.
        :param l1_reg:
        :param l2_reg:
        :param opt: Optimization function for the finetuning step.
        :param learning_rate: Learning rate for the finetuning.
        :param momentum: Momentum for the finetuning.
        :param num_epochs: Number of epochs for the finetuning.
        :param batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose:
        """

        super().__init__(name=name,
                         layers=layers,
                         dec_act_func=dec_act_func,
                         loss_func=loss_func,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, input_shape, n_output):

        """ Create the finetuning model
        :param input_shape:
        :param n_output:
        :return: self
        """

        fine_tuning_layers = []

        # Hidden layers
        for n, l in enumerate(self.layers):
            if isinstance(l, Autoencoder):
                fine_tuning_layers.append(Dense(output_dim=l.n_hidden,
                                                input_shape=[input_shape[1] if n == 0 else None],
                                                weights=l.get_model_parameters()['enc'],
                                                activation=l.enc_act_func,
                                                W_regularizer=l1l2(self.l1_reg, self.l2_reg),
                                                b_regularizer=l1l2(self.l1_reg, self.l2_reg)))
            else:
                fine_tuning_layers.append(l)

        # Output layer
        fine_tuning_layers.append(Dense(output_dim=n_output,
                                        activation=self.dec_act_func))
        return fine_tuning_layers

    def _pretrain(self, x_train, x_valid=None):

        """ Perform unsupervised pretraining of the stack of autoencoders.
        :param x_train: training set
        :param x_valid: validation set
        :return: self
        """

        self.logger.info('Starting {} unsupervised pretraining...'.format(self.name))

        next_train = x_train
        next_valid = x_valid

        for i, l in enumerate(self.layers):
            if isinstance(l, Autoencoder):
                self.logger.info('Pre-training layer {}'.format(i))

                l.fit(next_train, next_valid)

                # Encode the data for the next layer
                next_train = l.transform(data=next_train)

                if x_valid:
                    next_valid = l.transform(data=next_valid)

        self.logger.info('Done {} unsupervised pretraining...'.format(self.name))

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
