from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Dropout
from keras.regularizers import l1l2

from pydl.models.base.supervised_model import SupervisedModel


class MLP(SupervisedModel):

    """ Multi-Layer Perceptron
    """

    def __init__(self,
                 name='mlp',
                 layers=list([128, 64, 32]),
                 enc_act_func='relu',
                 dec_act_func='linear',
                 l1_reg=0.0,
                 l2_reg=0.0,
                 dropout=0.4,
                 loss_func='mse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 verbose=0,
                 seed=42):

        """
        :param name: Name of the model.
        :param layers: Number of hidden units in each layer.
        :param enc_act_func: Activation function for the hidden layers.
        :param dec_act_func: Activation function for the output layer.
        :param l1_reg: L1 weight regularization penalty, also known as LASSO.
        :param l2_reg: L2 weight regularization penalty, also known as weight decay, or Ridge.
        :param dropout: Fraction of the input units to drop. dropout = 1.0 (keep all).
        :param loss_func: Cost function.
        :param num_epochs: Number of training epochs.
        :param batch_size: Size of each training mini-batch.
        :param opt: Optimizer function.
        :param learning_rate: Initial learning rate.
        :param momentum: Initial momentum value.
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(name=name,
                         layers=layers,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
                         loss_func=loss_func,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         dropout=dropout,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, input_shape, n_output):

        """ Create the network layers
        :param input_shape:
        :param n_output:
        :return: self
        """

        for i, l in enumerate(self.layers + [n_output]):

            self._model.add(Dropout(p=self.dropout,
                                    input_shape=[input_shape[1] if i == 0 else None]))

            self._model.add(Dense(output_dim=l,
                                  activation=self.enc_act_func if i < len(self.layers) else self.dec_act_func,
                                  W_regularizer=l1l2(self.l1_reg, self.l2_reg),
                                  b_regularizer=l1l2(self.l1_reg, self.l2_reg)))
