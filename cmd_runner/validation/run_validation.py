import numpy as np
import tensorflow as tf
from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder
from pydl.models.autoencoder_models.stacked_denoising_autoencoder import StackedDenoisingAutoencoder
from pydl.models.nnet_models.mlp import MLP
from pydl.models.nnet_models.rnn import RNN
from pydl.utils import datasets
from pydl.utils.utilities import flag_to_list
from validator.model_validator import ModelValidator

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'mlp', 'Model to validate (use default parameters).')
flags.DEFINE_string('method', 'split', 'Validation method.')
flags.DEFINE_integer('n_folds', 10, 'Number of cv folds.')
flags.DEFINE_float('test_size', 0.3, 'Percentage of data used for validation.')
flags.DEFINE_string('dataset_x', '', 'Path to the dataset file (.npy or .csv).')
flags.DEFINE_string('dataset_y', '', 'Path to the dataset outputs file (.npy or .csv).')
flags.DEFINE_string('metrics', 'mse,mae,rmse', '')

model_name = FLAGS.model

params = {
  'method':    FLAGS.method,
  'k':         FLAGS.n_folds,
  'test_size': FLAGS.test_size,
  'dataset_x': FLAGS.dataset_x,
  'dataset_y': FLAGS.dataset_y,
  'metrics':   flag_to_list(FLAGS.metrics, 'str'),
}


def run_validation(model, **kwargs):

    dataset = datasets.load_dataset(kwargs.get('dataset_x'),
                                    kwargs.get('dataset_y'))

    valid = ModelValidator(method=kwargs.get('method'),
                           k=kwargs.get('k'),
                           test_size=kwargs.get('test_size'))

    res = valid.run(model=model,
                    x=dataset.data,
                    y=dataset.target,
                    metrics=kwargs.get('metrics'))

    print('\n::::::::::::::::::::::::::::::::::::::::::::')
    print('::           VALIDATION RESULTS           ::')
    print('::::::::::::::::::::::::::::::::::::::::::::\n')
    print('> Test score = %.4f (std dev = %.4f)' % (np.mean(res['scores']), np.std(res['scores'])))

    for m, v in res['metrics'].items():
        print('> %s = %.4f (std dev = %.4f)' % (m, np.mean(v), np.std(v)))


if __name__ == '__main__':

    if model_name == 'mlp':
        m = MLP()
    elif model_name == 'rnn':
        m = RNN(cell_type='simple')
    elif model_name == 'lstm':
        m = RNN(cell_type='lstm')
    elif model_name == 'gru':
        m = RNN(cell_type='gru')
    elif model_name == 'sae':
        m = StackedAutoencoder()
    elif model_name == 'sdae':
        m = StackedDenoisingAutoencoder()
    else:
        raise Exception('Invalid model!')

    run_validation(m, **params)
