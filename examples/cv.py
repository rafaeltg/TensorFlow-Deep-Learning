import json
from pydl.datasets import mackey_glass, create_dataset
from pydl.models import MLP
from pydl.model_selection import CV


def run_cv():

    """
        Cross-Validation examples
    """

    print('Creating dataset')
    # Create time series data
    ts = mackey_glass(sample_len=2000)
    x, y = create_dataset(ts, look_back=10)

    print('Creating MLP')
    mlp = MLP(layers=[32, 16],
              dropout=0.1,
              nb_epochs=100)

    print('Creating TrainTestSplitCV method')
    cv = CV(method='split', test_size=0.2)

    print('Running CV!')
    res = cv.run(model=mlp, x=x, y=y, scoring=['mape', 'rmse'])

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))

    print('\nCreating TimeSeriesCV method')
    cv = CV(method='time_series', window=1100, horizon=100, by=100, fixed=False)

    print('Running CV!')
    res = cv.run(model=mlp, x=x, y=y, scoring=['mape', 'rmse'], max_threads=4)

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))

if __name__ == '__main__':
    run_cv()
