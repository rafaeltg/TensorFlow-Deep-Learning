from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder
from pydl.models.autoencoder_models.stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from pydl.models.nnet_models.mlp import MLP
from pydl.models.nnet_models.rnn import RNN


def fit(config):
    print('fit', config)


def predict(config):
    print('predict', config)


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


def get_model(config):
    assert 'model' in config, 'Missing model definition'
    assert 'class' in config['model'], 'Missing model class'

    m = config['model']['class']
    p = config['model']['params'] if 'params' in config['model'] else {}

    if m == 'mlp':
        return MLP(**p)
    elif m == 'rnn':
        return RNN(**p)
    elif m == 'sae':
        return StackedAutoencoder(**p)
    elif m == 'sdae':
        return StackedDenoisingAutoencoder(**p)
    else:
        raise Exception('Invalid model!')
