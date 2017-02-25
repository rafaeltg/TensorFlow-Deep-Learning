from keras.layers import Dense, Dropout
from keras.regularizers import l1l2

from .autoencoder import Autoencoder
from ..base import SupervisedModel


class StackedAutoencoder(SupervisedModel):

    """ Implementation of Stacked Autoencoders.
    """

    def __init__(self,
                 name='sae',
                 layers=None,
                 activation='relu',
                 out_activation='linear',
                 dropout=0,
                 **kwargs):

        """
        :param layers: List the Autoencoders
        """

        super().__init__(name=name,
                         layers=layers,
                         activation=activation,
                         out_activation=out_activation,
                         dropout=dropout,
                         **kwargs)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert all([isinstance(l, Autoencoder) for l in self.layers]), 'Layers must be Autoencoders'

    def _create_layers(self, input_shape, n_output):

        """ Create the finetuning model
        :param input_shape:
        :param n_output:
        :return: self
        """

        # Hidden layers
        for i, l in enumerate(self.layers):
            self._model.add(Dense(input_shape=[input_shape[1] if i == 0 else None],
                                  output_dim=l.n_hidden,
                                  weights=l.get_model_parameters()['enc'],
                                  activation=l.enc_activation,
                                  W_regularizer=l1l2(l.l1_reg, l.l2_reg),
                                  b_regularizer=l1l2(l.l1_reg, l.l2_reg)))

            if self.dropout[i] > 0:
                self._model.add(Dropout(p=self.dropout[i]))

        # Output layer
        self._model.add(Dense(output_dim=n_output, activation=self.out_activation))

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
