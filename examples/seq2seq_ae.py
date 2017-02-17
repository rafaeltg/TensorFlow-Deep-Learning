import os

import numpy as np
from pydl.models import SeqToSeqAutoencoder
from pydl.utils.utilities import load_model


def run_seq2seq_ae():

    """
        Sequence-to-Sequence Autoencoder example
    """

    # Create fake dataset
    input_dim = 20
    train_size = 2000
    test_size = 200
    time_steps = 2
    x_train = np.random.normal(loc=0, scale=1, size=(train_size, time_steps, input_dim))
    x_test = np.random.normal(loc=0, scale=1, size=(test_size, time_steps, input_dim))

    # reshape input to be [n_samples, time_steps, n_features]

    print('Creating Seq2Seq Autoencoder')
    hidden_size = 15
    s2s_ae = SeqToSeqAutoencoder(n_hidden=hidden_size,
                                 time_steps=time_steps,
                                 num_epochs=100)

    print('Training')
    s2s_ae.fit(x_train=x_train)

    train_score = s2s_ae.score(data=x_train)
    print('Reconstruction loss for training dataset = {}'.format(train_score))

    test_score = s2s_ae.score(data=x_test)
    print('Reconstruction loss for test dataset = {}'.format(test_score))

    print('Transforming data')
    x_test_tr = s2s_ae.transform(data=x_test)
    print('Transformed data shape = {}'.format(x_test_tr.shape))
    assert x_test_tr.shape == (test_size, hidden_size)

    print('Reconstructing data')
    x_test_rec = s2s_ae.reconstruct(x_test_tr)
    print('Reconstructed data shape = {}'.format(x_test_rec.shape))
    assert x_test_rec.shape == x_test.shape

    print('Saving model')
    s2s_ae.save_model('models/', 's2s_ae')
    assert os.path.exists('models/s2s_ae.json')
    assert os.path.exists('models/s2s_ae.h5')

    print('Loading model')
    s2s_ae_new = load_model('models/s2s_ae.json')

    print('Transforming data')
    x_test_tr_new = s2s_ae_new.transform(data=x_test)
    assert np.array_equal(x_test_tr, x_test_tr_new)

    print('Reconstructing data')
    x_test_rec_new = s2s_ae_new.reconstruct(x_test_tr_new)
    assert np.array_equal(x_test_rec, x_test_rec_new)

    print('Calculating training set score')
    train_score_new = s2s_ae_new.score(data=x_train)
    assert train_score == train_score_new

    print('Calculating testing set score')
    test_score_new = s2s_ae_new.score(data=x_test)
    assert test_score == test_score_new


if __name__ == '__main__':
    run_seq2seq_ae()
