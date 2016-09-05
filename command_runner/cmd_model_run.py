import numpy as np

from models.base.supervised_model import SupervisedModel
from models.base.unsupervised_model import UnsupervisedModel
from utils import datasets


#########################
#  Execution functions  #
#########################

def run_supervised_model(model, global_params):

    """
    :param model:
    :param global_params:
    :return: self
    """

    assert isinstance(model, SupervisedModel)
    assert global_params['train_dataset'] != '', 'Missing training dataset!'
    assert global_params['train_labels'] != '', 'Missing training labels!'

    # Read dataset
    data = datasets.load_datasets(train_dataset=global_params['train_dataset'],
                                  train_labels=global_params['train_labels'],
                                  test_dataset=global_params['test_dataset'],
                                  test_labels=global_params['test_labels'],
                                  valid_dataset=global_params['valid_dataset'],
                                  valid_labels=global_params['valid_labels'],
                                  has_header=True)

    train_x = data.train.data
    train_y = data.train.target
    valid_x = data.validation.data
    valid_y = data.validation.target
    test_x  = data.test.data
    test_y  = data.test.target

    # Train the model
    model.fit(train_x, train_y, valid_x, valid_y)

    if test_x is not None:
        test_cost = model.evaluate(test_x, test_y)
        print('\nTest Cost = {}'.format(test_cost))

        # Save the predictions of the model
        if global_params['save_predictions']:
            np.save(global_params['save_predictions'], model.predict(test_x))


def run_unsupervised_model(model, global_params):

    """
    :param model:
    :param global_params:
    :return: self
    """

    assert isinstance(model, UnsupervisedModel)
    assert global_params['train_dataset'] != '', 'Missing training dataset!'

    # Read dataset
    data = datasets.load_datasets(train_dataset=global_params['train_dataset'],
                                  test_dataset=global_params['test_dataset'],
                                  valid_dataset=global_params['valid_dataset'],
                                  has_header=True)

    train_x = data.train.data
    valid_x = data.validation.data
    test_x  = data.test.data

    # Train the model
    model.fit(train_x, valid_x)

    if global_params['save_encode_train']:
        np.save(global_params['save_encode_train'], model.transform(train_x))

    if (valid_x is not None) and global_params['save_encode_valid']:
        np.save(global_params['save_encode_valid'], model.transform(valid_x))

    if test_x is not None:
        test_cost = model.evaluate(data=test_x)
        print('\nTest Cost = {}'.format(test_cost))

        if global_params['save_encode_test']:
            np.save(global_params['save_encode_test'], model.transform(test_x))