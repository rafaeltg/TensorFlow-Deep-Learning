from pydl.datasets.utils import load_data_file


def get_model(config):
    assert 'model' in config, 'Missing model definition!'
    return config['model']


def get_input_data(config):
    assert 'data_set' in config, 'Missing data set path!'
    return config['data_set']


def load_data(data_set, name):
    data_config = data_set.get(name, None)
    assert data_config is not None, 'Missing "%s" input!' % name
    assert 'path' in data_config and data_config['path'] != '', 'Missing file path for %s' % name
    params = data_config.get('params', {}) if 'params' in data_config else {}
    return load_data_file(data_config['path'], **params)


def get_cv_config(config):
    assert 'cv' in config, 'Missing cross-validation configurations!'
    cv_config = config['cv']
    assert 'method' in cv_config, 'Missing cross-validation method!'
    method = cv_config['method']
    params = cv_config['params'] if 'params' in cv_config else {}
    scoring = cv_config['scoring'] if 'scoring' in cv_config else None
    max_threads = cv_config['max_threads'] if 'max_threads' in cv_config else 1
    return method, params, scoring, max_threads
