from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN

import pydl.utils.utilities as utils
from pydl.models.base.supervised_model import SupervisedModel


class RNN(SupervisedModel):

    """ Generic Recurrent Neural Network
    """

    def __init__(self,
                 model_name='rnn',
                 main_dir='rnn/',
                 cell_type='lstm',
                 layers=list([50, 50]),
                 stateful=True,
                 enc_act_func='tanh',
                 dec_act_func='linear',
                 loss_func='mse',
                 num_epochs=100,
                 batch_size=1,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 dropout=0.2,
                 verbose=0,
                 seed=42):

        """
        :param model_name: Name of the model.
        :param main_dir: Directory to save the model data.
        :param cell_type: Recurrent layers type. ["lstm", "gru", "simple"]
        :param layers: Number of hidden units in each layer.
        :param stateful: Whether the recurrent network is stateful or not.It means that the states
            computed for the samples in one batch will be reused as initial states for the samples
            in the next batch.
        :param enc_act_func: Activation function for the hidden layers.
        :param dec_act_func: Activation function for the output layer.
        :param loss_func: Cost function.
        :param num_epochs: Number of training epochs.
        :param batch_size: Size of each training mini-batch.
        :param opt: Optimizer function.
        :param learning_rate: Initial learning rate.
        :param momentum: Initial momentum value.
        :param dropout: The probability that each element is kept at each layer. Default = 1.0 (keep all).
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         loss_func=loss_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert cell_type in ["lstm", "gru", "simple"], 'Invalid cell type!'
        assert len(layers) > 0
        assert all([l > 0 for l in layers])
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions
        assert 0 <= dropout <= 1.0

        if cell_type == 'lstm':
            self.cell = LSTM
        elif cell_type == 'gru':
            self.cell = GRU
        else:
            self.cell = SimpleRNN

        self.stateful = stateful
        self.layers = layers
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.dropout = dropout

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, input_shape, n_output):

        """ Create the network layers
        :param n_output:
        :return: self
        """

        b_size = self.batch_size if self.stateful else None

        # Hidden layers
        for i, l in enumerate(self.layers):

            self._model.add(Dropout(p=self.dropout,
                                    batch_input_shape=(b_size, input_shape[1], input_shape[2]) if i == 0 else [None]))

            self._model.add(self.cell(output_dim=l,
                                      batch_input_shape=(b_size, input_shape[1], input_shape[2]),
                                      activation=self.enc_act_func,
                                      stateful=self.stateful,
                                      return_sequences=True if i < (len(self.layers)-1) else False))

        # Output layer
        self._model.add(Dropout(p=self.dropout))
        self._model.add(Dense(output_dim=n_output,
                              activation=self.dec_act_func))

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):

        x_train = self._check_shape(x_train)

        if x_valid:
            x_valid = self._check_shape(x_valid)

        super().fit(x_train, y_train, x_valid, y_valid)

    def _train_step(self, x_train, y_train, x_valid=None, y_valid=None):

        if self.stateful:

            for i in range(self.num_epochs):
                if self.verbose > 0:
                    print('>> Epoch', i, '/', self.num_epochs)

                self._model.fit(x=x_train,
                                y=y_train,
                                batch_size=self.batch_size,
                                verbose=self.verbose,
                                nb_epoch=1,
                                shuffle=False,
                                validation_data=(x_valid, y_valid) if x_valid and y_valid else None)
                self._model.reset_states()
        else:

            super()._train_step(x_train, y_train, x_valid, y_valid)

    def predict(self, data):

        data = self._check_shape(data)

        return super().predict(data)

    def score(self, x, y):

        x = self._check_shape(x)

        return super().score(x, y)

    def _check_shape(self, data):

        if len(data.shape) != 3:
            data = np.reshape(data, (data.shape[0], 1, data.shape[1]))

        return data
