import os
import sys
import json
import inspect
import keras.models as k_models


valid_act_functions = ['softmax', 'softplus', 'sigmoid', 'tanh', 'relu', 'linear']

valid_loss_functions = ['mse',                       # Mean Squared Error
                        'mae',                       # Mean Absolute Error
                        'mape',                      # Mean Absolute Percentage Error
                        'msle',                      # Mean Squared Logarithmic Error
                        'binary_crossentropy',       # Log loss
                        'categorical_crossentropy',  # Multiclass Log loss
                        'kld'                        # Kullback Leibler Divergence (information gain)
                        ]

valid_opt_functions = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam']


def expand_arg(layers, arg_to_expand):

    """Expands the arg_to_expand into the length of layers.
    This is used as a convenience so that the user does not need to specify the
    complete list of parameters for model initialization.
    IE: the user can just specify one parameter and this function will expand it
    :param layers:
    :param arg_to_expand:
    :return:
    """

    if not isinstance(arg_to_expand, list):
        arg_to_expand = [arg_to_expand]

    if len(arg_to_expand) == len(layers):
        return arg_to_expand

    if len(arg_to_expand) > len(layers):
        return arg_to_expand[0:len(layers)]

    missing_values = len(layers) - len(arg_to_expand)
    result = arg_to_expand + [arg_to_expand[-1] for _ in range(missing_values)]

    return result


def save_model(model, dir=None, file_name=None):
    if dir is None:
        dir = os.getcwd()
    elif not os.path.exists(dir):
        os.makedirs(dir)

    if file_name is None:
        file_name = model.name

    cfg = model.to_json()

    if model.is_built():
        cfg['weights'] = os.path.join(dir, file_name + '.h5')
        k_models.save_model(getattr(model, '_model'), cfg['weights'])

    save_json(cfg, os.path.join(dir, file_name+'.json'))


def load_model(config=None):
    assert config is not None, 'Missing input configuration'

    configs = config
    if isinstance(config, str):
        configs = load_json(config)

    assert len(configs) > 0, 'No configuration specified!'
    assert 'model' in configs, 'Missing model definition!'

    m = model_from_config(configs['model'])
    if m is None:
        raise Exception('Invalid model!')

    if 'weights' in configs:
        m.load_model(configs['weights'])

    return m


def model_from_config(config):
    assert 'class_name' in config, 'Missing model class!'

    # fetch all members of module 'pydl.models'
    classes = dict(inspect.getmembers(sys.modules['pydl.models'], inspect.isclass))
    return k_models.model_from_config(config, classes)


def load_json(inp):
    if os.path.isfile(inp):
        with open(inp) as file:
            data = json.load(file)
    else:
        data = json.loads(inp)

    return data


def save_json(data, file_path, sort_keys=True):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=4, ensure_ascii=False)



