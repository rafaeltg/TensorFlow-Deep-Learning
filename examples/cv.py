import json

from sklearn.preprocessing import MinMaxScaler

from examples.synthetic import mackey_glass, create_dataset
from pydl.models import MLP
from pydl.validator.model_validator import ModelValidator

from keras.layers import Dense, Dropout


def run_cv():

    """
        Cross-Validation examples
    """

    print('Creating dataset')
    # Create time series data
    ts = mackey_glass(sample_len=2000)
    # Normalize the dataset
    ts = MinMaxScaler(feature_range=(0, 1)).fit_transform(ts)
    x, y = create_dataset(ts, look_back=10)

    print('Creating MLP')
    mlp = MLP(
        layers=[
            Dense(input_shape=[10],
                  output_dim=32,
                  activation='relu'),
            Dropout(p=0.1),
            Dense(output_dim=16,
                  activation='relu'),
            Dropout(p=0.1)
        ],
        num_epochs=100,
    )

    print('Creating TrainTestSplitCV method')
    cv = ModelValidator(method='split', test_size=0.2)

    print('Running CV!')
    res = cv.run(model=mlp, x=x, y=y, metrics=['mape', 'rmse'])

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))

    print('\nCreating TimeSeriesCV method')
    cv = ModelValidator(method='time_series', window=1000, horizon=100, by=100, fixed=False)

    print('Running CV!')
    res = cv.run(model=mlp, x=x, y=y, metrics=['mape', 'rmse'])

    print('CV results:')
    print(json.dumps(res, indent=4, separators=(',', ': ')))

if __name__ == '__main__':
    run_cv()
