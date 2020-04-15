import os
import numpy as np
from pydl.model_selection.scorer import r2_score, rmse
from pydl.models.layers import Dense, Dropout
from pydl.models import MLP, load_model, model_from_json
from dataset import create_multivariate_data, train_test_split


def run_mlp():

    """
        MLP example
    """

    n_features = 4

    data = create_multivariate_data(n_features=n_features)
    x, y, x_test, y_test = train_test_split(data[:, :-1], data[:, -1])

    print('Creating MLP')
    model = MLP(
        name='mlp',
        layers=[
            Dense(units=16, activation='relu'),
            Dropout(0.1),
            Dense(units=8, activation='relu')
        ],
        epochs=200)

    print('Training')
    model.fit(x=x, y=y)

    print(model.summary())

    train_score = model.score(x=x, y=y)
    print('Train {} = {}'.format(model.get_loss_func().upper(), train_score))

    test_score = model.score(x=x_test, y=y_test)
    print('Test {} = {}'.format(model.get_loss_func().upper(), test_score))

    print('Predicting test data')
    y_test_pred = model.predict(x_test)

    y_test_rmse = rmse(y_test, y_test_pred)
    print('y_test RMSE = {}'.format(y_test_rmse))

    y_test_r2 = r2_score(y_test, y_test_pred)
    print('y_test R2 = {}'.format(y_test_r2))

    print('Saving model')
    model.save('models/mlp.h5')
    model.save_json('models/mlp.json')
    model.save_weights('models/mlp_weights.h5')

    assert os.path.exists('models/mlp.json')
    assert os.path.exists('models/mlp.h5')
    assert os.path.exists('models/mlp_weights.h5')

    del model

    print('Loading model from .h5 file')
    model = load_model('models/mlp.h5')
    assert isinstance(model, MLP)
    assert model.name == 'mlp'

    print('Calculating train score')
    np.testing.assert_equal(train_score, model.score(x=x, y=y))

    print('Calculating test score')
    np.testing.assert_equal(test_score, model.score(x=x_test, y=y_test))

    print('Predicting test data')
    y_test_pred_new = model.predict(x_test)
    np.testing.assert_allclose(y_test_pred, y_test_pred_new, atol=1e-6)

    print('Calculating RMSE for test set')
    np.testing.assert_equal(y_test_rmse, rmse(y_test, y_test_pred_new))

    print('Calculating R2 for test set')
    np.testing.assert_equal(y_test_r2, r2_score(y_test, y_test_pred_new))

    del model

    print('Loading model from json and weights files')
    model = model_from_json('models/mlp.json', weights_filepath='models/mlp_weights.h5', compile=True)
    assert isinstance(model, MLP)
    assert model.name == 'mlp'

    print('Calculating train score')
    np.testing.assert_equal(train_score, model.score(x=x, y=y))

    print('Calculating test score')
    np.testing.assert_equal(test_score, model.score(x=x_test, y=y_test))

    print('Predicting test data')
    y_test_pred_new = model.predict(x_test)
    np.testing.assert_allclose(y_test_pred, y_test_pred_new, atol=1e-6)

    print('Calculating RMSE for test set')
    np.testing.assert_equal(y_test_rmse, rmse(y_test, y_test_pred_new))

    print('Calculating R2 for test set')
    np.testing.assert_equal(y_test_r2, r2_score(y_test, y_test_pred_new))


if __name__ == '__main__':
    run_mlp()
