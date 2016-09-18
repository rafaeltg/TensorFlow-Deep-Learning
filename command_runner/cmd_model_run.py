import numpy as np
import matplotlib.pyplot as plt
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
        test_score = model.score(test_x, test_y)
        print('\nTest Loss = {}'.format(test_score))

        preds = model.predict(test_x)
        plot_predictions(test_y, preds)

        # Save the predictions of the model
        if global_params['save_predictions']:
            np.save(global_params['save_predictions'], preds)


def plot_predictions(y, y_pred, p=1):

    ax = plt.subplot(111)
    ax.plot(y[1:int((len(y)*p))], label='Actual')
    ax.plot(y_pred[1:int((len(y_pred)*p))], label='Predicted')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
    plt.show()


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
        test_loss = model.score(data=test_x)
        print('\nTest Loss = {}'.format(test_loss))

        if global_params['save_encode_test']:
            np.save(global_params['save_encode_test'], model.transform(test_x))
