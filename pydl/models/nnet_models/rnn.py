import numpy as np
from keras.layers import Dense, LSTM, GRU, SimpleRNN
from ..base import SupervisedModel
from ..utils import expand_arg

_cell_type = {
    'lstm': LSTM,
    'gru': GRU,
    'simple': SimpleRNN
}


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
                 dropout=0,
                 recurrent_dropout=0,
                 implementation=0,
                 **kwargs):

        """
        :param name: Name of the model.
        :param layers: Number of hidden units in each layer.
        :param stateful: Whether the recurrent network is stateful or not.It means that the states
            computed for the samples in one batch will be reused as initial states for the samples
            in the next batch.
        :param time_steps:
        :param cell_type: Recurrent layers type. ["lstm", "gru", "simple"]
        :param dropout:
        :param recurrent_dropout:
        :param implementation: one of {0, 1, or 2}.
            - 0 = the RNN will use an implementation that uses fewer, larger matrix products, thus running 
            faster on CPU but consuming more memory.
            - 1 = the RNN will use more matrix products, but smaller ones, thus running slower (may actually be 
            faster on GPU) while consuming less memory.
            - 2 (LSTM/GRU only) = the RNN will combine the input gate, the forget gate and the output gate into 
            a single matrix, enabling more time-efficient parallelization on the GPU.
            Note: RNN dropout must be shared for all gates, resulting in a slightly reduced regularization.
        """

        self.stateful = stateful
        self.time_steps = time_steps
        self.cell_type = cell_type
        self.recurrent_dropout = expand_arg(layers, recurrent_dropout)
        self.implementation = implementation

        super().__init__(name=name,
                         layers=layers,
                         activation=activation,
                         out_activation=out_activation,
                         dropout=dropout,
                         **kwargs)

    def validate_params(self):
        super().validate_params()
        assert self.time_steps > 0, "time_steps must be grater than zero!"
        assert self.cell_type in ["lstm", "gru", "simple"], 'Invalid cell type!'
        assert all([0 <= d <= 1 for d in self.recurrent_dropout]), 'Invalid recurrent_dropout value!'
        assert self.implementation in [0, 1, 2], 'Invalid implementation mode'

    def _create_layers(self, input_shape, n_output):

        """ Create the network layers
        :param input_shape:
        :param n_output:
        :return: self
        """

        cell = _cell_type[self.cell_type]
        b_size = self.batch_size if self.stateful else None

        # Hidden layers
        for i, l in enumerate(self.layers):
            self._model.add(cell(units=l,
                                 batch_input_shape=(b_size, input_shape[1], input_shape[2]),
                                 activation=self.activation[i],
                                 stateful=self.stateful,
                                 return_sequences=True if i < (len(self.layers)-1) else False,
                                 dropout=self.dropout[i],
                                 recurrent_dropout=self.recurrent_dropout[i],
                                 implementation=self.implementation))

        # Output layer
        self._model.add(Dense(units=n_output, activation=self.out_activation))

    def _check_x_shape(self, x):
        if len(x.shape) == 2:
            x = np.reshape(x, (x.shape[0], self.time_steps, x.shape[1]))
        return x

    def _train_step(self, x_train, y_train, valid_data=None, valid_split=0.):
        if self.stateful:
            for cb in self._callbacks:
                cb.set_model(self._model.model.callback_model)
                cb.on_train_begin()

            last = 0
            for i in range(self.nb_epochs):
                print('%d / %d' % (i, self.nb_epochs))
                logs = self._model.fit(x=x_train,
                                       y=y_train,
                                       batch_size=self.batch_size,
                                       epochs=1,
                                       shuffle=False,
                                       validation_data=valid_data,
                                       validation_split=valid_split,
                                       verbose=self.verbose)
                self._model.reset_states()

                print(logs.history['val_loss'][0])
                print(last - logs.history['val_loss'][0])
                last = logs.history['val_loss'][0]

                for cb in self._callbacks:
                    cb.on_epoch_end(epoch=i, logs={'val_loss': logs.history['val_loss'][0]})

                if self._model.model.callback_model.stop_training:
                    break

        else:
            super()._train_step(x_train, y_train, valid_data, valid_split)
