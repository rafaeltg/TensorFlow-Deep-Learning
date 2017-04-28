import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pydl.datasets import mackey_glass, create_dataset
from pydl.model_selection import r2_score
from pydl.models import RNN
from pydl.models.utils import load_model, save_model


def run_lstm():

    """
        LSTM example
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

    # reshape input to be [n_samples, time_steps, n_features]
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    print('Creating a stateless LSTM')
    lstm = RNN(layers=[50, 50],
               dropout=[0.1, 0.2],
               stateful=False,
               time_steps=1,
               cell_type='lstm',
               nb_epochs=300,
               batch_size=100)

    print('Training')
    lstm.fit(x_train=x_train, y_train=y_train)

    train_score = lstm.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = lstm.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = lstm.predict(x_test)

    assert y_test_pred.shape == y_test.shape

    y_test_r2 = r2_score(y_test, y_test_pred)
    print('r2_score for y_test forecasting = {}'.format(y_test_r2))

    print('Saving model')
    save_model(lstm, 'models/', 'lstm')
    assert os.path.exists('models/lstm.json')
    assert os.path.exists('models/lstm.h5')

    print('Loading model')
    lstm_new = load_model('models/lstm.json')

    print('Calculating train score')
    assert train_score == lstm_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == lstm_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = lstm_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating r2 score')
    assert y_test_r2 == r2_score(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_lstm()
