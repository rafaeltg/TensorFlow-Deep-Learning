import json

from sklearn.preprocessing import MinMaxScaler

from examples.synthetic import mackey_glass, create_dataset
from pydl.hyperopt import *
from pydl.model_selection.methods import TrainTestSplitCV


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
    space = HyperOptSpace({
        'model': {
            'class_name': 'MLP',
            'config': {
                'layers': [hp_int(10, 1000), hp_int(10, 1000)],
                'dropout': [hp_float(0, 0.3), hp_float(0, 0.3)],
                'activation': hp_choice(['relu', 'tanh', 'sigmoid'])
            }
        }
    })

    print('Creating CV Method')
    cv = TrainTestSplitCV(test_size=0.2)

    print('Creating CMAES optimizer')
    opt = CMAESOptimizer(pop_size=10, max_iter=20)

    print('Creating HyperOptModel...')
    model = HyperOptModel(hp_space=space, cv=cv, opt=opt)

    print('Optimizing!')
    res = model.fit(x, y)

    print('Best parameters:')
    best_params = res['best_model_config']
    print(json.dumps(best_params, indent=4, separators=(',', ': ')))

    print('Test RMSE of the best model = {}'.format(res['opt_result'][1]))


if __name__ == '__main__':
    run_optimizer()
