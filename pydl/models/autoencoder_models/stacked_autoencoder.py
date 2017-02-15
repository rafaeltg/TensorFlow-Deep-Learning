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
                 **kwargs):

        """
        :param layers: List the Autoencoders
        """

        super().__init__(name=name,
                         layers=layers,
                         **kwargs)

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
                                                W_regularizer=l1l2(l.l1_reg, l.l2_reg),
                                                b_regularizer=l1l2(l.l1_reg, l.l2_reg)))
            else:
                fine_tuning_layers.append(l)

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
