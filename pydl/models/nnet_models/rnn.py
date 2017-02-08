from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Recurrent

from ..base import SupervisedModel


class RNN(SupervisedModel):

    """ Generic Recurrent Neural Network
    """

    def __init__(self,
                 name='rnn',
                 layers=None,
                 stateful=True,
                 time_steps=1,
                 dec_act_func='linear',
                 loss_func='mse',
                 num_epochs=100,
                 batch_size=1,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 verbose=0,
                 seed=42):

        """
        :param name: Name of the model.
        :param layers: Number of hidden units in each layer.
        :param stateful: Whether the recurrent network is stateful or not.It means that the states
            computed for the samples in one batch will be reused as initial states for the samples
            in the next batch.
        :param time_steps:
        :param dec_act_func: Activation function for the output layer.
        :param loss_func: Cost function.
        :param num_epochs: Number of training epochs.
        :param batch_size: Size of each training mini-batch.
        :param opt: Optimizer function.
        :param learning_rate: Initial learning rate.
        :param momentum: Initial momentum value.
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        self.stateful = stateful
        self.time_steps = time_steps

        super().__init__(name=name,
                         layers=layers,
                         dec_act_func=dec_act_func,
                         loss_func=loss_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def validate_params(self):
        super().validate_params()
        assert self.time_steps > 0, "time_steps must be grater than zero!"

    def _create_layers(self, input_shape, n_output):
        assert len(input_shape) == 3, 'Invalid input shape'

        if isinstance(self.layers[0], Recurrent) and not hasattr(self.layers[0], 'batch_input_shape'):
            b_size = self.batch_size if self.stateful else None
            self.layers[0].batch_input_shape = (b_size, self.time_steps, input_shape[2])

        # Check return_sequences and stateful parameters
        last_layer = True
        for l in self.layers[::-1]:
            if isinstance(l, Recurrent):
                l.stateful = self.stateful
                if last_layer:
                    l.return_sequences = False
                    last_layer = False
                else:
                    l.return_sequences = True

        # Add output layer
        self.add_output_layer(n_output)
        return self.layers

    def _train_step(self, x_train, y_train, x_valid=None, y_valid=None):

        if self.stateful:
            for i in range(self.num_epochs):
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
