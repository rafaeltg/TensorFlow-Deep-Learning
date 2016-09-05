from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import utils.utilities as utils
from models.base.supervised_model import SupervisedModel
from models.nnet_models.hidden_layer import HiddenLayer


class MLP(SupervisedModel):

    """ Multi-Layer Perceptron
    """

    def __init__(self,
                 model_name='mlp',
                 main_dir='mlp/',
                 layers=list([128, 64, 32]),
                 enc_act_func='tanh',
                 dec_act_func='none',
                 cost_func='rmse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 dropout=1.0,
                 verbose=0,
                 seed=-1,
                 task='regression'):

        """
        :param model_name: Name of the model.
        :param main_dir: Directory to save the model data.
        :param layers: Number of hidden units in each layer.
        :param enc_act_func: Activation function for the hidden layers. ['tanh', 'sigmoid', 'relu', 'linear']
        :param dec_act_func: Activation function for the output layer. ['tanh', 'sigmoid', 'relu', 'linear']
        :param cost_func: Cost function. ['rmse', 'cross_entropy', 'softmax_cross_entropy']
        :param num_epochs: Number of training epochs.
        :param batch_size: Size of each training mini-batch.
        :param opt: Optimizer function. ['gradient_descent', 'momentum', 'ada_grad', 'adam', 'rms_prop']
        :param learning_rate: Initial learning rate.
        :param momentum: Initial momentum value.
        :param dropout: The probability that each element is kept at each layer. Default = 1.0 (keep all).
        :param verbose: Level of verbosity. 0 - silent, 1 - print everything.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param task: ['regression', 'classification']
        """

        super().__init__(model_name,
                         main_dir,
                         cost_func,
                         num_epochs,
                         batch_size,
                         opt,
                         learning_rate,
                         momentum,
                         seed,
                         verbose,
                         task)

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Validations
        assert len(layers) > 0
        assert all([l > 0 for l in layers])
        assert enc_act_func in utils.valid_act_functions
        assert dec_act_func in utils.valid_act_functions
        assert cost_func in utils.valid_supervised_cost_functions
        assert 0 <= dropout <= 1.0

        self.layers = layers
        self.enc_act_func = enc_act_func
        self.dec_act_func = dec_act_func
        self.dropout = dropout

        self._layer_nodes = []
        self._model_predictions = None

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_input, n_output):

        """ Create the network layers
        :param n_input:
        :param n_output:
        :return: self
        """

        layer_inp = self._input
        self._layer_nodes = []

        for l, layer in enumerate(self.layers):
            layer_node = HiddenLayer(input_layer=layer_inp,
                                     hidden_units=layer,
                                     act_func=self.enc_act_func,
                                     dropout=self.dropout,
                                     name_scope='mlp_layer_{}'.format(l))

            layer_inp = layer_node.get_output()

            self._layer_nodes.append(layer_node)

        self._create_output_layer(n_output, layer_inp)

    def _create_output_layer(self, n_output, input_layer):

        """ Create the final layer for finetuning.
        :param n_output: size of the output layer
        :param input_layer:
        :return: self
        """

        out_layer = HiddenLayer(input_layer=input_layer,
                                hidden_units=n_output,
                                act_func=self.dec_act_func,
                                name_scope='mlp_output_layer')

        self._layer_nodes.append(out_layer)

        self._model_output = out_layer.get_output()

        with tf.name_scope("model_predictions"):
            if self.task == 'classification':
                self.model_predictions = tf.argmax(self._model_output, 1)

            else:
                self.model_predictions = self._model_output

    def _train_model(self, train_set, train_labels, valid_set, valid_labels):

        """ Train the model.
        :param train_set: training set
        :param train_labels: training labels
        :param valid_set:
        :param valid_labels:
        :return: self
        """

        self.logger.info('Training {}'.format(self.model_name))

        n_out = train_labels.shape[1]

        shuff = list(zip(train_set, train_labels))

        for i in range(self.num_epochs):
            self.logger.info('Training epoch {}...'.format(i))

            np.random.shuffle(shuff)
            batches = [_ for _ in utils.gen_batches(shuff, self.batch_size)]

            for batch in batches:
                x_batch, y_batch = list(zip(*batch))
                y_batch = np.reshape(y_batch, (-1, n_out))
                self.tf_session.run(self.optimizer, feed_dict={self._input: x_batch,
                                                               self._target_output: y_batch})

            if valid_set is not None:
                self._run_monitor(i, {self._input: valid_set, self._target_output: valid_labels})

        self.logger.info('Done Training {}'.format(self.model_name))

    def get_model_parameters(self, graph=None):

        """ Return the model parameters in the form of numpy arrays.
        :param graph: tf graph object
        :return: model parameters
        """

        g = graph if graph is not None else self.tf_graph

        params = []

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)

                for _, layer in enumerate(self._layer_nodes):
                    params.append({
                        'w': layer.get_weights().eval(),
                        'b': layer.get_biases().eval()
                    })

        return params

    def get_layers_output(self, dataset):

        """ Get output from each layer of the network.
        :param dataset: input data
        :return: list of np array, element i in the list is the output of layer i
        """

        layers_out = []

        with tf.Session() as self.tf_session:
            self.tf_saver.restore(self.tf_session, self.model_path)
            for l in self._layer_nodes:
                layers_out.append(l.get_output().eval({self._input: dataset}))

        return layers_out
