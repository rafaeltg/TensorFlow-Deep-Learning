import json
import os
import numpy as np

from optimizer.optimizer import CMAESOptimizer
from optimizer.parameter_dictionary import ParameterDictionary
from pydl.models.autoencoder_models.autoencoder import Autoencoder
from pydl.models.autoencoder_models.denoising_autoencoder import DenoisingAutoencoder
from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder
from pydl.models.autoencoder_models.stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from pydl.models.base.supervised_model import SupervisedModel
from pydl.models.base.unsupervised_model import UnsupervisedModel
from pydl.models.nnet_models.mlp import MLP
from pydl.models.nnet_models.rnn import RNN
from pydl.utils import datasets
from validator.cv_methods import get_cv_method
from validator.model_validator import ModelValidator
from validator.cv_metrics import available_metrics


def fit(config):
    """
    """

    print('fit', config)

    m = load_model(config)

    data_set = get_input_data(config)
    x = load_data(data_set, 'train_x')

    if isinstance(m, SupervisedModel):
        y = load_data(data_set, 'train_y')
        m.fit(x_train=x, y_train=y)
    else:
        m.fit(x_train=x)

    # Save model
    m.save_model(config['output'])


def predict(config):
    """
    """

    print('predict', config)

    m = load_model(config)
    assert isinstance(m, SupervisedModel), 'The given model cannot perform predict operation'

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    preds = m.predict(x)
    # Save predictions as .npy file
    np.save(os.path.join(config['output'], m.name+'_preds.npy'), preds)


def transform(config):
    """
    """

    print('transform', config)

    m = load_model(config)
    assert isinstance(m, UnsupervisedModel), 'The given model cannot perform transform operation'

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    x_encoded = m.transform(x)

    # Save encoded data in a .npy file
    base_name = os.path.splitext(os.path.basename(data_set['data_x']))[0]
    np.save(os.path.join(config['output'], base_name+'_encoded.npy'), x_encoded)


def reconstruct(config):
    """
    """

    print('reconstruct', config)

    m = load_model(config)
    assert isinstance(m, UnsupervisedModel), 'The given model cannot perform reconstruct operation'

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    x_rec = m.reconstruct(x)

    # Save reconstructed data in a .npy file
    base_name = os.path.splitext(os.path.basename(data_set['data_x']))[0]
    np.save(os.path.join(config['output'], base_name+'_reconstructed.npy'), x_rec)


def score(config):
    """
    """

    print('score', config)

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    metrics = config['metrics'] if 'metrics' in config else []

    results = {}

    m = load_model(config)
    if isinstance(m, SupervisedModel):
        y = load_data(data_set, 'data_y')

        results['score'] = m.score(x, y)
        if len(metrics) > 0:
            y_pred = m.predict(x)
            for metric in metrics:
                results[metric] = available_metrics[metric](y, y_pred)

    else:
        results['score'] = m.score(x)
        if len(metrics) > 0:
            x_rec = m.reconstruct(m.transform(x))
            for metric in metrics:
                results[metric] = available_metrics[metric](x, x_rec)

    # Save results into a JSON file
    save_json(results, os.path.join(config['output'], m.name+'_scores.json'))


def validate(config):
    """
    """
    print('validate', config)

    m = load_model(config)

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')
    y = load_data(data_set, 'data_y')

    # Get validation method
    method, params, metrics = get_cv_config(config)
    cv = ModelValidator(method=method, **params)
    results = cv.run(model=m, x=x, y=y, metrics=metrics)

    # Save results into a JSON file
    save_json(results, os.path.join(config['output'], m.name+'_cv.json'))


def optimize(config):
    """
    :param config:
    :return:
    """
    print('optimize', config)

    m = load_model(config)

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')
    y = load_data(data_set, 'data_y') if isinstance(m, SupervisedModel) else None

    assert 'params' in config, "Missing List of parameters to optimize"
    params = ParameterDictionary()
    params.from_json(config['params'])

    method, params, _ = get_cv_config(config)
    cv_method = get_cv_method(method=method, **params)

    fit_fn = available_metrics[config['cv']['metric']]

    opt = get_optimizer(config, cv_method, fit_fn)

    result = opt.run(model=m,
                     params_dict=params,
                     x=x,
                     y=y,
                     max_thread=4)

    best_params = params.get(result[0])
    print('best params =', best_params)

    # fit the model with the best parameters using all data
    m.set_params(**best_params)
    if isinstance(m, SupervisedModel):
        m.fit(x_train=x, y_train=y)
    else:
        m.fit(x_train=x)

    # Save model
    m.save_model(config['output'])


#
# UTILS
#
def load_model(config):
    assert 'model' in config, 'Missing model definition!'
    model_config = config['model']

    m = get_model_by_class(model_config)

    if 'params' in model_config:
        m.set_params(**model_config['params'])
    elif 'path' in model_config:
        m.load_model(model_config['path'])

    return m


def get_model_by_class(model_config):
    assert 'class' in model_config, 'Missing model class!'
    c = model_config['class']

    if c == 'mlp':
        return MLP()
    elif c == 'rnn':
        return RNN()
    elif c == 'sae':
        return StackedAutoencoder()
    elif c == 'sdae':
        return StackedDenoisingAutoencoder()
    elif c == 'ae':
        return Autoencoder()
    elif c == 'dae':
        return DenoisingAutoencoder()
    else:
        raise Exception('Invalid model!')


def get_input_data(config):
    assert 'data_set' in config, 'Missing data set path!'
    return config['data_set']


def get_cv_config(config):
    assert 'cv' in config, 'Missing cross-validation configurations!'
    cv_config = config['cv']
    assert 'method' in cv_config, 'Missing cross-validation method!'
    method = cv_config['method']
    params = cv_config['params'] if 'params' in cv_config else {}
    metrics = config['cv']['metrics'] if 'metrics' in config['cv'] else []
    return method, params, metrics


def get_optimizer(config, cv, fit_fn):
    assert 'opt' in config, 'Missing optimizer parameters!'
    opt_config = config['opt']
    assert 'method' in opt_config, 'Missing optimization method'
    method = opt_config['method']
    params = opt_config['params'] if 'params' in opt_config else {}

    if method == 'cmaes':
        return CMAESOptimizer(cv=cv, fit_fn=fit_fn, **params)
    else:
        raise AttributeError('Invalid optimizer method')


def load_data(data_set, set_name):
    data_path = data_set.get(set_name, None)
    assert data_path is not None and data_path != '', 'Missing %s file!' % set_name
    has_header = data_set.get('has_header', False)
    return datasets.load_data_file(data_path, has_header=has_header)


def save_json(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=False, indent=4, ensure_ascii=False)
