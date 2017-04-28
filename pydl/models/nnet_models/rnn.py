import numpy as np
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from ..base import SupervisedModel


class RNN(SupervisedModel):

    """ Generic Recurrent Neural Network
    """

    def __init__(self,
                 name='rnn',
                 layers=None,
                 stateful=False,
                 time_steps=1,
                 cell_type='lstm',
                 activation='tanh',
                 out_activation='linear',
                 **kwargs):

        """
        :param name: Name of the model.
        :param layers: Number of hidden units in each layer.
        :param stateful: Whether the recurrent network is stateful or not.It means that the states
            computed for the samples in one batch will be reused as initial states for the samples
            in the next batch.
        :param time_steps:
        :param cell_type: Recurrent layers type. ["lstm", "gru", "simple"]
        """

        self.stateful = stateful
        self.time_steps = time_steps
        self.cell_type = cell_type

        super().__init__(name=name,
                         layers=layers,
                         activation=activation,
                         out_activation=out_activation,
                         **kwargs)

    def validate_params(self):
        super().validate_params()
        assert self.time_steps > 0, "time_steps must be grater than zero!"
        assert self.cell_type in ["lstm", "gru", "simple"], 'Invalid cell type!'

    def _create_layers(self, input_shape, n_output):

        """ Create the network layers
        :param n_output:
        :return: self
        """

        if self.cell_type == 'lstm':
            cell = LSTM
        elif self.cell_type == 'gru':
            cell = GRU
        else:
            cell = SimpleRNN

        b_size = self.batch_size if self.stateful else None

        # Hidden layers
        for i, l in enumerate(self.layers):

            self._model.add(cell(units=l,
                                 batch_input_shape=(b_size, input_shape[1], input_shape[2]),
                                 activation=self.activation[i],
                                 stateful=self.stateful,
                                 return_sequences=True if i < (len(self.layers)-1) else False))

            if self.dropout[i] > 0:
                self._model.add(Dropout(rate=self.dropout[i]))

        # Output layer
        self._model.add(Dense(units=n_output, activation=self.out_activation))

    def _check_x_shape(self, x):
        if len(x.shape) == 2:
            x = np.reshape(x, (x.shape[0], self.time_steps, x.shape[1]))
        return x

    def _train_step(self, x_train, y_train, valid_data=None, valid_split=0.):

        if self.stateful:
            for i in range(self.nb_epochs):
                print('>> Epoch', i, '/', self.nb_epochs)

                self._model.fit(x=x_train,
                                y=y_train,
                                batch_size=self.batch_size,
                                epochs=1,
                                shuffle=False,
                                validation_data=valid_data,
                                verbose=self.verbose)
                self._model.reset_states()
        else:
            super()._train_step(x_train, y_train, valid_data, valid_split)
