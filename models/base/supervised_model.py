from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils.utilities as utils
from models.base.model import Model


class SupervisedModel(Model):

    """ Class representing an abstract Supervised Model.
    """

    def __init__(self,
                 model_name,
                 main_dir,
                 loss_func='rmse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 seed=-1,
                 verbose=0,
                 task='regression'):

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         loss_func=loss_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose,
                         task=task)

        # Validation
        assert loss_func in utils.valid_supervised_cost_functions

        # Supervised Model output values placeholder
        self._target_output = None

    def build_model(self, n_input, n_output=1):

        """ Creates the computational graph for the Supervised Model.
        :param n_input: Number of features.
        :param n_output: number of output values.
        :return: self
        """

        self.logger.info('Building {} model'.format(self.model_name))

        self._create_placeholders(n_input, n_output)
        self._create_layers(n_input, n_output)

        self._create_cost_node(self._target_output)
        self._create_optimizer_node()

        self.logger.info('Done building {} model'.format(self.model_name))

    def _create_placeholders(self, n_input, n_output):

        """ Create the TensorFlow placeholders for the Supervised Model.
        :param n_input: number of features of the first layer
        :param n_output: size of the output layer
        :return: self
        """

        self._input = tf.placeholder('float', [None, n_input], name='x-input')
        self._target_output = tf.placeholder('float', [None, n_output], name='y-input')

    def _create_layers(self, n_input, n_output):
        pass

    def fit(self, train_set, train_labels, valid_set, valid_labels, restore_previous_model=False, graph=None):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features)
        :param train_labels: Training labels. shape(n_samples, n_classes)
        :param valid_set:
        :param valid_labels:
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        :param graph: TensorFlow graph object
        :return: self
        """

        self.logger.info('Starting {} supervised training...'.format(self.model_name))

        if len(train_labels.shape) != 1:
            num_out = train_labels.shape[1]
        else:
            self.logger.error('Invalid training labels shape')
            raise Exception("Please convert the labels with one-hot encoding.")

        g = graph if graph is not None else self.tf_graph

        with g.as_default():

            self.build_model(train_set.shape[1], num_out)

            with tf.Session() as self.tf_session:
                self._initialize_tf(restore_previous_model)
                self._train_model(train_set, train_labels, valid_set, valid_labels)
                self.tf_saver.save(self.tf_session, self.model_path)

        self.logger.info('Done {} supervised training...'.format(self.model_name))

    def _train_model(self, train_set, train_labels, valid_set, valid_labels):
        pass

    def predict(self, data):

        """ Predict the labels for the test set.
        :param data: Testing data. shape(n_test_samples, n_features)
        :return: labels
        """

        with self.tf_graph.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                preds = self.model_predictions.eval({self._input: data})

        return preds

    def calc_total_cost(self, data, data_labels, graph=None):

        """ Compute the total reconstruction cost.
        :param data: Input data
        :param data_labels:
        :param graph: TensorFlow graph object
        :return: reconstruction cost
        """

        g = graph if graph is not None else self.tf_graph
        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                cost = self.cost.eval({self._input: data,
                                       self._target_output: data_labels})

        return cost
