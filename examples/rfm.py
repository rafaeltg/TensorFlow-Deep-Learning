import os
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pydl.datasets import mackey_glass, create_dataset
from pydl.model_selection import r2_score
from pydl.models import RNN, RFM
from pydl.models.utils import load_model, save_model
from pydl.hyperopt import *


def run_fem():

    """
        FEM example
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
               dropout=[0., 0.2],
               stateful=False,
               time_steps=1,
               cell_type='lstm',
               nb_epochs=100,
               batch_size=50)

    fem = RFM(fm=lstm, em=lstm.copy())

    print('Training')
    fem.fit(x_train=x_train, y_train=y_train)

    #train_score = lstm.score(x=x_train, y=y_train)
    #print('Train score = {}'.format(train_score))

    #test_score = lstm.score(x=x_test, y=y_test)
    #print('Test score = {}'.format(test_score))

    print('Predicting test data')
    y_test_pred = fem.predict(x_test)
    assert y_test_pred.shape == y_test.shape

    y_test_r2 = r2_score(y_test, y_test_pred)
    print('r2_score for y_test forecasting = {}'.format(y_test_r2))

    print('Saving model')
    save_model(fem, 'models/', 'fem')
    assert os.path.exists('models/fem.json')
    assert os.path.exists('models/fem_fm.h5')
    assert os.path.exists('models/fem_em.h5')

    print('Loading model')
    fem_new = load_model('models/fem.json')

    #print('Calculating train score')
    #assert train_score == lstm_new.score(x=x_train, y=y_train)

    #print('Calculating test score')
    #assert test_score == lstm_new.score(x=x_test, y=y_test)

    print('Predicting test data')
    y_test_pred_new = fem_new.predict(x_test)
    assert np.array_equal(y_test_pred, y_test_pred_new)

    print('Calculating r2 score')
    assert y_test_r2 == r2_score(y_test, y_test_pred_new)


def optimize_fem():

    """
        Optimizing FEM
    """

    print('Creating dataset')
    # Create time series data
    ts = mackey_glass(sample_len=1000)
    # Normalize the dataset
    ts = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts)
    x, y = create_dataset(ts, look_back=10)

    # reshape input to be [n_samples, time_steps, n_features]
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

    print('Creating FEM ConfigOptimizer')
    space = hp_space({
        'model': {
            'class_name': 'FEM',
            'config': {
                'name': 'fem',
                'fm': {
                    'model': {
                        'class_name': 'RNN',
                        'config': {
                            'name': 'lstm_fm',
                            'cell_type': 'lstm',
                            'layers': [hp_int(10, 100)],
                            'dropout': [hp_float(0, 0.5)],
                            'stateful': False,
                        },
                    }
                },
                'em': {
                    'model': {
                        'class_name': 'RNN',
                        'config': {
                            'name': 'lstm_fm',
                            'cell_type': 'lstm',
                            'layers': [hp_int(10, 100)],
                            'dropout': [hp_float(0, 0.5)],
                            'stateful': False,
                        },
                    }
                }
            }
        }
    })

    print('Creating Fitness Function')
    fit_fn = CVObjectiveFunction(scoring='r2')

    print('Creating HyperOptModel...')
    m = HyperOptModel(hp_space=space, fit_fn=fit_fn, opt='cmaes', opt_args={'pop_size': 8, 'max_iter': 1})

    print('Optimizing!')
    res = m.fit(x, y, max_threads=4)

    print('Best parameters:')
    best_params = res['best_model_config']
    print(json.dumps(best_params, indent=4, separators=(',', ': ')))

    print('Test R2 of the best model = {}'.format(res['opt_result'][1]))

if __name__ == '__main__':
    #run_fem()
    optimize_fem()
