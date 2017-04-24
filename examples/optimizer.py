import json
from sklearn.preprocessing import MinMaxScaler
from pydl.datasets import mackey_glass, create_dataset
from pydl.hyperopt import *


def run_optimizer():

    """
        CMAES Optimizer example
    """

    print('Creating dataset')
    # Create time series data
    ts = mackey_glass(sample_len=2000)
    # Normalize the dataset
    ts = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts)
    x, y = create_dataset(ts, look_back=10)

    print('Creating MLP ConfigOptimizer')
    space = hp_space({
        'model': {
            'class_name': 'MLP',
            'config': hp_choice([
                {
                    'name': 'mlp_',
                    'layers': [hp_int(10, 512)],
                    'dropout': hp_float(0, 0.5),
                    'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                    'nb_epochs': hp_int(100, 200),
                    'batch_size': hp_int(32, 512),
                    'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                    'learning_rate': hp_float(0.0001, 0.01)
                },
                {
                    'name': 'mlp_',
                    'layers': [hp_int(10, 512), hp_int(10, 512)],
                    'dropout': [hp_float(0, 0.5), hp_float(0, 0.5)],
                    'activation': hp_choice(['relu', 'tanh', 'sigmoid']),
                    'nb_epochs': hp_int(100, 200),
                    'batch_size': hp_int(32, 512),
                    'opt': hp_choice(['adam', 'rmsprop', 'adadelta']),
                    'learning_rate': hp_float(0.0001, 0.01)
                },
            ])
        }
    })

    print('Creating Fitness Function')
    fit_fn = CVObjectiveFunction(scoring='mse')

    print('Creating HyperOptModel...')
    m = HyperOptModel(hp_space=space, fit_fn=fit_fn, opt='cmaes', opt_args={'pop_size': 8, 'max_iter': 4})

    print('Optimizing!')
    res = m.fit(x, y, max_threads=4)

    print('Best parameters:')
    best_params = res['best_model_config']
    print(json.dumps(best_params, indent=4, separators=(',', ': ')))

    print('Test RMSE of the best model = {}'.format(res['opt_result'][1]))


if __name__ == '__main__':
    run_optimizer()