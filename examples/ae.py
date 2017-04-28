import os
import numpy as np
from pydl.models.autoencoder_models import Autoencoder
from pydl.models.utils import load_model, save_model


def run_ae():

    """
        Autoencoder example
    """

    # Create fake dataset
    input_dim = 20
    train_size = 2000
    test_size = 200
    x_train = np.random.normal(loc=0, scale=1, size=(train_size, input_dim))
    x_test = np.random.normal(loc=0, scale=1, size=(test_size, input_dim))

    print('Creating Autoencoder')
    hidden_size = 15
    ae = Autoencoder(
        n_hidden=hidden_size,
        nb_epochs=100
    )

    print('Training')
    ae.fit(x_train=x_train)

    train_score = ae.score(data=x_train)
    print('Reconstruction loss for training dataset = {}'.format(train_score))

    test_score = ae.score(data=x_test)
    print('Reconstruction loss for test dataset = {}'.format(test_score))

    print('Transforming data')
    x_test_tr = ae.transform(data=x_test)
    print('Transformed data shape = {}'.format(x_test_tr.shape))
    assert x_test_tr.shape == (test_size, hidden_size)

    print('Reconstructing data')
    x_test_rec = ae.reconstruct(x_test_tr)
    print('Reconstructed data shape = {}'.format(x_test_rec.shape))
    assert x_test_rec.shape == x_test.shape

    print('Saving model')
    save_model(ae, 'models/', 'ae')
    assert os.path.exists('models/ae.json')
    assert os.path.exists('models/ae.h5')

    print('Loading model')
    ae_new = load_model('models/ae.json')

    print('Transforming data')
    x_test_tr_new = ae_new.transform(data=x_test)
    assert np.array_equal(x_test_tr, x_test_tr_new)

    print('Reconstructing data')
    x_test_rec_new = ae_new.reconstruct(x_test_tr_new)
    assert np.array_equal(x_test_rec, x_test_rec_new)

    print('Calculating training set score')
    train_score_new = ae_new.score(data=x_train)
    assert train_score == train_score_new

    print('Calculating testing set score')
    test_score_new = ae_new.score(data=x_test)
    assert test_score == test_score_new

if __name__ == '__main__':
    run_ae()
