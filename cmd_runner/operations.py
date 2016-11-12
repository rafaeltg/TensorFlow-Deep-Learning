from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder
from pydl.models.autoencoder_models.stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from pydl.models.nnet_models.mlp import MLP
from pydl.models.nnet_models.rnn import RNN
from pydl.models.base.supervised_model import SupervisedModel
from pydl.utils import datasets


def fit(config):
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
    print('predict', config)

    m = get_model(config)

    data_set = get_input_data(config)
    x = load_data(data_set, 'data_x')

    preds = m.predict(x)
    print(preds)


def transform(config):
    print('transform', config)


def reconstruct(config):
    print('reconstruct', config)


def score(config):
    print('score', config)


def validate(config):
    print('validate', config)


def optimize(config):
    print('optimize', config)


def load_model(config):
    assert 'model' in config, 'Missing model definition!'
    model_config = config['model']

    print('Model config ===>', model_config)

    m = get_model(model_config)

    if 'params' in model_config:
        m.set_params(**model_config['params'])
    elif 'path' in model_config:
        m.load_model(model_config['path'])
    else:
        raise Exception('Missing model configurations!')

    return m


def get_model(model_config):
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
    else:
        raise Exception('Invalid model!')


def get_input_data(config):
    assert 'data_set' in config, 'Missing data set path!'
    return config['data_set']


def load_data(data_set, set_name):
    data_path = data_set.get(set_name, None)
    assert data_path is not None and data_path != '', 'Missing %s file!' % set_name
    has_header = data_set.get('has_header', False)
    return datasets.load_data_file(data_path, has_header=has_header)
