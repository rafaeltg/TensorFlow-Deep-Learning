import os
import json
import importlib
import numpy as np
from keras.utils.layer_utils import layer_from_config

valid_act_functions = ['softmax', 'softplus', 'sigmoid', 'tanh', 'relu', 'linear']

valid_loss_functions = ['mse',                       # Mean Squared Error
                        'mae',                       # Mean Absolute Error
                        'mape',                      # Mean Absolute Percentage Error
                        'msle',                      # Mean Squared Logarithmic Error
                        'binary_crossentropy',       # Log loss
                        'categorical_crossentropy',  # Multiclass Log loss
                        'kld'                        # Kullback Leibler Divergence (information gain)
                        ]

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

    m = model_from_config(configs['model'])
    if m is None:
        raise Exception('Invalid model!')

    if 'weights' in configs:
        m.load_model(configs['weights'])

    return m


def model_from_config(config):
    assert 'class_name' in config, 'Missing model class!'
    class_name = config['class_name']
    model_config = config['config'] if 'config' in config else {}

    models = importlib.import_module('.models', 'pydl')
    if hasattr(models, class_name):
        model_class = getattr(models, class_name)
        return model_class.from_config(model_config)

    return None


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


def layers_from_config(layers_config):
    layers = []
    for l in layers_config:
        layer = model_from_config(l)
        if layer is None:
            layer = layer_from_config(l)
        layers.append(layer)
    return layers



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
