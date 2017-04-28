import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pydl.datasets import mackey_glass, create_dataset
from pydl.model_selection.metrics import mape
from pydl.models import MLP
from pydl.models.utils import load_model, save_model


def run_mlp():

    """
        MLP example
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

    print('Creating MLP')
    mlp = MLP(layers=[32, 16],
              activation='relu',
              out_activation='linear',
              dropout=0.1,
              l1_reg=0.00001,
              l2_reg=0.00001,
              nb_epochs=400,
              early_stopping=True,
              patient=2,
              min_delta=1e-4)

    print('Training')
    mlp.fit(x_train=x_train, y_train=y_train)

    train_score = mlp.score(x=x_train, y=y_train)
    print('Train score = {}'.format(train_score))

    test_score = mlp.score(x=x_test, y=y_test)
    print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = mlp.predict(x_test)
    print('Predicted y_test shape = {}'.format(y_test_pred.shape))
    assert y_test_pred.shape == y_test.shape

    y_test_mape = mape(y_test, y_test_pred)
    print('MAPE for y_test forecasting = {}'.format(y_test_mape))

    print('Saving model')
    save_model(mlp, 'models/', 'mlp')
    assert os.path.exists('models/mlp.json')
    assert os.path.exists('models/mlp.h5')

    print('Loading model')
    mlp_new = load_model('models/mlp.json')

    print('Calculating train score')
    assert train_score == mlp_new.score(x=x_train, y=y_train)

    print('Calculating test score')
    assert test_score == mlp_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = mlp_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating MAPE')
    assert y_test_mape == mape(y_test, y_test_pred_new)


if __name__ == '__main__':
    run_mlp()
