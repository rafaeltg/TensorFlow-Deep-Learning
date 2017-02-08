import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from examples.synthetic import mackey_glass, create_dataset
from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder
from pydl.models.autoencoder_models.autoencoder import Autoencoder
from pydl.validator.cv_metrics import mape
from pydl.utils.utilities import load_model


def run_sae():

    """
        Stacked Autoencoder example
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

    print('Creating Stacked Autoencoder')
    sae = StackedAutoencoder(
        layers=[
            Autoencoder(n_hidden=32, enc_act_func='relu'),
            Autoencoder(n_hidden=16, enc_act_func='relu'),
        ],
    )

    print('Training')
    sae.fit(x_train=x_train, y_train=y_train)

    train_score = sae.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = sae.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = sae.predict(data=x_test)
    print('Predicted y_test shape = {}'.format(y_test_pred.shape))
    assert y_test_pred.shape == y_test.shape

    y_test_mape = mape(y_test, y_test_pred)
    print('MAPE for y_test forecasting = {}'.format(y_test_mape))

    print('Saving model')
    sae.save_model('/home/rafael/models/', 'sae')
    assert os.path.exists('/home/rafael/models/sae.json')
    assert os.path.exists('/home/rafael/models/sae.h5')

    print('Loading model')
    sae_new = load_model('/home/rafael/models/sae.json')

    print('Calculating train score')
    assert train_score == sae_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == sae_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = sae_new.predict(data=x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating MAPE')
    assert y_test_mape == mape(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_sae()
