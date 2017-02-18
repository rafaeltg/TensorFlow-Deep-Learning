import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from examples.synthetic import mackey_glass, create_dataset
from pydl.models import StackedAutoencoder, DenoisingAutoencoder
from pydl.model_selection.metrics import mape
from pydl.utils.utilities import load_model
from keras.layers import Dense


def run_sdae():

    """
        Stacked Denoising Autoencoder example
    """

    # Create time series data
    ts = mackey_glass(sample_len=2000)
    # Normalize the dataset
    ts = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts)

    # split into train and test sets
    train_size = int(len(ts) * 0.8)
    train, test = ts[0:train_size], ts[train_size:len(ts)]

    # reshape into X=t and Y=t+1
    look_back = 10
    x_train, y_train = create_dataset(train, look_back)
    x_test, y_test = create_dataset(test, look_back)

    print('Creating Stacked Denoising Autoencoder')
    sdae = StackedAutoencoder(
        layers=[DenoisingAutoencoder(n_hidden=32,
                                     enc_act_func='relu',
                                     corr_type='masking',
                                     corr_param=0.1),
                DenoisingAutoencoder(n_hidden=16,
                                     enc_act_func='relu',
                                     corr_type='masking',
                                     corr_param=0.1)
        ],
        num_epochs=100
    )

    print('Training')
    sdae.fit(x_train=x_train, y_train=y_train)

    train_score = sdae.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = sdae.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = sdae.predict(x_test)
    print('Predicted y_test shape = {}'.format(y_test_pred.shape))
    assert y_test_pred.shape == y_test.shape

    y_test_mape = mape(y_test, y_test_pred)
    print('MAPE for y_test forecasting = {}'.format(y_test_mape))

    print('Saving model')
    sdae.save_model('models/', 'sdae')
    assert os.path.exists('models/sdae.json')
    assert os.path.exists('models/sdae.h5')

    print('Loading model')
    sdae_new = load_model('models/sdae.json')

    print('Calculating train score')
    assert train_score == sdae_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == sdae_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = sdae_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating MAPE')
    assert y_test_mape == mape(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_sdae()
