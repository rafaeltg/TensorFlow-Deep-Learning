import json

from sklearn.preprocessing import MinMaxScaler

from examples.synthetic import mackey_glass, create_dataset
from pydl.optimizer.optimizer import CMAESOptimizer
from pydl.optimizer.config_optimizer import *
from pydl.model_selection.cv_methods import TrainTestSplitCV
from pydl.model_selection.cv_metrics import rmse


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
    conf = ConfigOptimizer(
        class_name='MLP',
        config={
            'layers': ListParameter([
                {
                    'class_name': 'Dense',
                    'config': {
                        'output_dim': IntegerParameter(32, 64),
                        'activation': CategoricalParameter(['tanh', 'linear', 'relu']),
                    }
                },
                {
                    'class_name': 'Dropout',
                    'config': {
                        'p': RealParameter(0, 0.2),
                    }
                },
                {
                    'class_name': 'Dense',
                    'config': {
                        'output_dim': IntegerParameter(16, 32),
                        'activation': CategoricalParameter(['tanh', 'linear', 'relu']),
                    }
                },
                {
                    'class_name': 'Dropout',
                    'config': {
                        'p': RealParameter(0, 0.2),
                    }
                },
            ]),
            'num_epochs': IntegerParameter(100, 300),
        }
    )

    print('Creating CV Method')
    cv = TrainTestSplitCV(test_size=0.2)

    print('Creating CMAES optimizer')
    opt = CMAESOptimizer(cv=cv, fit_fn=rmse, pop_size=10, max_iter=20)

    print('Optimizing!')
    res = opt.run(
        config_opt=conf,
        x=x,
        y=y,
        max_thread=4,
    )

    print('Best parameters:')
    best_params = conf.get_best_config(res[0])
    print(json.dumps(best_params, indent=4, separators=(',', ': ')))

    print('Test RMSE of the best model = {}'.format(res[1]))


if __name__ == '__main__':
    run_optimizer()
