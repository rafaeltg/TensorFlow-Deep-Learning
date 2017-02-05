import os
import json
import numpy as np


valid_act_functions = ['softmax', 'softplus', 'sigmoid', 'tanh', 'relu', 'linear']

valid_loss_functions = ['mse',                       # Mean Squared Error
                        'mae',                       # Mean Absolute Error
                        'mape',                      # Mean Absolute Percentage Error
                        'msle',                      # Mean Squared Logarithmic Error
                        'binary_crossentropy',       # Log loss
                        'categorical_crossentropy',  # Multiclass Log loss
                        'kld',                       # Kullback Leibler Divergence (information gain)
                        'custom']

valid_optimization_functions = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam']


# ################ #
#   Data helpers   #
# ################ #

def gen_batches(data, batch_size):

    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    data = np.array(data)

    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]


# ############# #
#   Utilities   #
# ############# #

def load_model(config=None):
    assert config is not None, 'Missing input configuration'

    configs = config
    if isinstance(config, str):
        configs = load_json(config)

    assert len(configs) > 0, 'No configuration specified!'
    assert 'model' in configs, 'Missing model definition!'
    model = configs['model']

    assert 'class_name' in model, 'Missing model class!'
    config = model['config'] if 'config' in model else {}

    m = build_model(model['class_name'], config)

    if 'weights' in configs:
        m.load_weights(configs['weights'])

    return m


def build_model(m, config):
    import importlib
    import pkgutil

    for model_module in ['autoencoder_models', 'nnet_models']:
        mod = '.models.' + model_module
        module = importlib.import_module(mod, 'pydl')
        pkgpath = os.path.dirname(module.__file__)
        for _, name, _ in pkgutil.iter_modules([pkgpath]):
            class_mod = mod + '.' + name
            class_module = importlib.import_module(class_mod, 'pydl')
            if hasattr(class_module, m):
                c = getattr(class_module, m)
                return c.from_config(config)

    raise Exception('Invalid model!')


def load_json(inp):
    if os.path.isfile(inp):
        with open(inp) as file:
            data = json.load(file)
    else:
        data = json.loads(inp)

    return data


def save_json(data, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, sort_keys=False, indent=4, ensure_ascii=False)


def create_dir(dir_path):

    """
    :param dir_path: directory to be created
    """

    try:
        if not os.path.exists(dir_path):
            print('Creating %s directory.' % dir_path)
            os.makedirs(dir_path)
    except OSError as e:
        raise e


def expand_args(layers, args_to_expand):

    """Expands all the lists in args_to_expand into the length of layers.
    This is used as a convenience so that the user does not need to specify the
    complete list of parameters for model initialization.
    IE: the user can just specify one parameter and this function will expand it
    :param layers:
    :param args_to_expand:
    :return:
    """

    for key, val in args_to_expand.items():
        if isinstance(val, list) and (len(val) != len(layers)):
            args_to_expand[key] = [val[0] for _ in layers]

    return args_to_expand


def flag_to_list(flag_val, dtype):

    """
    :param flag_val:
    :param dtype:
    :return:
    """

    if dtype == 'int':
        return [int(_) for _ in flag_val.split(',') if _]

    elif dtype == 'float':
        return [float(_) for _ in flag_val.split(',') if _]

    elif dtype == 'str':
        return [_ for _ in flag_val.split(',') if _]

    else:
        raise Exception("Incorrect data type")
