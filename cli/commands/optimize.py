from .utils import get_input_data, load_data, get_cv_config
from pydl.hyperopt import HyperOptModel, hp_space_from_json, CVObjectiveFunction


def optimize(config, output):
    """
    """

    # Get hp_space
    assert 'hp_space' in config, 'Missing hyperparameters space definition'
    space = hp_space_from_json(config['hp_space'])

    # Get data
    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')
    y = load_data(data_set, 'data_y') if 'data_y' in data_set else None

    # Get HyperOptModel
    assert 'opt' in config, 'Missing optimizer parameters!'
    opt_config = config['opt']
    opt, opt_params = get_optimizer(opt_config)
    obj_fn = get_obj_fn(opt_config)
    max_threads = opt_config['max_threads'] if 'max_threads' in opt_config else 1
    retrain = opt_config['retrain'] if 'retrain' in opt_config else False

    opt_model = HyperOptModel(hp_space=space, fit_fn=obj_fn, opt=opt, opt_args=opt_params)
    opt_model.fit(x, y, retrain=retrain, max_threads=max_threads)

    # Save best model
    opt_model.save_model(output)


def get_optimizer(config):
    assert 'method' in config, 'Missing optimization method'
    method = config['method']
    m = method['class']
    params = method['params'] if 'params' in method else {}
    return m, params


def get_obj_fn(config):
    if 'obj_fn' in config:
        obj_fn_config = config['obj_fn']

        if 'cv' in obj_fn_config:
            method, params, scoring, _ = get_cv_config(obj_fn_config)
            return CVObjectiveFunction(scoring=scoring, cv_method=method, **params)

    return None
