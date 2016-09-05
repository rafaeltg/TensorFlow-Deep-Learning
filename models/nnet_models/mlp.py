from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils.utilities as utils
from keras.layers import Dense, Dropout
from models.base.supervised_model import SupervisedModel


class MLP(SupervisedModel):

    """ Multi-Layer Perceptron
    """

    def __init__(self,
                 model_name='mlp',
                 main_dir='mlp/',
                 layers=list([128, 64, 32]),
                 enc_act_func='relu',
                 dec_act_func='linear',
                 loss_func='mean_squared_error',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 dropout=0.4,
                 verbose=0,
                 seed=-1):

        """
        :param model_name: Name of the model.
        :param main_dir: Directory to save the model data.
        :param layers: Number of hidden units in each layer.
        :param enc_act_func: Activation function for the hidden layers. ['tanh', 'sigmoid', 'relu', 'linear']
        :param dec_act_func: Activation function for the output layer. ['tanh', 'sigmoid', 'relu', 'linear']
        :param loss_func: Cost function. ['mean_squared_error', 'cross_entropy', 'softmax_cross_entropy']
        :param num_epochs: Number of training epochs.
        :param batch_size: Size of each training mini-batch.
        :param opt: Optimizer function. ['gradient_descent', 'momentum', 'ada_grad', 'adam', 'rms_prop']
        :param learning_rate: Initial learning rate.
        :param momentum: Initial momentum value.
        :param dropout: The probability that each element is kept at each layer. Default = 1.0 (keep all).
        :param verbose: Level of verbosity. 0 - silent, 1 - print everything.
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
        assert len(layers) > 0
        assert all([l > 0 for l in layers])
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions
        assert 0 <= dropout <= 1.0

        self.layers = layers
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.dropout = dropout

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_input, n_output):

        """ Create the network layers
        :param n_input:
        :param n_output:
        :return: self
        """

        # Hidden layers
        for n, l in enumerate(self.layers):
            self._model.add(Dense(output_dim=l,
                                  init='uniform',
                                  activation=self.enc_act_func,
                                  input_dim=n_input if n == 0 else None))

            if self.dropout < 1:
                self._model.add(Dropout(p=self.dropout))

        # Output layer
        self._model.add(Dense(output_dim=n_output,
                              activation=self.dec_act_func))

    def get_model_parameters(self):

        """ Return the model parameters in the form of numpy arrays.
        :return: model parameters
        """

        params = [l.get_weigths for l in self._model.layers]

        return params
