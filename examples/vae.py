import os
import numpy as np
from pydl.models import VariationalAutoencoder
from pydl.models.utils import load_model, save_model


def run_vae():

    """
        Variational Autoencoder example
    """

    # Create fake dataset
    input_dim = 20
    train_size = 2000
    test_size = 200
    x_train = np.random.normal(loc=0, scale=1, size=(train_size, input_dim))
    x_test = np.random.normal(loc=0, scale=1, size=(test_size, input_dim))

    print('Creating Variational Autoencoder')
    n_hidden = 15
    n_latent = 5
    vae = VariationalAutoencoder(
        n_latent=n_latent,
        n_hidden=n_hidden,
        nb_epochs=100
    )

    print('Training')
    vae.fit(x_train=x_train)

    train_score = vae.score(data=x_train)
    print('Reconstruction loss for training dataset = {}'.format(train_score))

    test_score = vae.score(data=x_test)
    print('Reconstruction loss for test dataset = {}'.format(test_score))

    print('Transforming data')
    x_test_tr = vae.transform(data=x_test)
    print('Transformed data shape = {}'.format(x_test_tr.shape))
    assert x_test_tr.shape == (test_size, n_latent)

    print('Reconstructing data')
    x_test_rec = vae.reconstruct(x_test_tr)
    print('Reconstructed data shape = {}'.format(x_test_rec.shape))
    assert x_test_rec.shape == x_test.shape

    print('Saving model')
    save_model(vae, 'models/', 'vae')
    assert os.path.exists('models/vae.json')
    assert os.path.exists('models/vae.h5')

    print('Loading model')
    vae_new = load_model('models/vae.json')

    print('Transforming data')
    x_test_tr_new = vae_new.transform(data=x_test)
    assert np.array_equal(x_test_tr, x_test_tr_new)

    print('Reconstructing data')
    x_test_rec_new = vae_new.reconstruct(x_test_tr_new)
    assert np.array_equal(x_test_rec, x_test_rec_new)

    print('Calculating training set score')
    train_score_new = vae_new.score(data=x_train)
    assert train_score == train_score_new

    print('Calculating testing set score')
    test_score_new = vae_new.score(data=x_test)
    assert test_score == test_score_new

if __name__ == '__main__':
    run_vae()
