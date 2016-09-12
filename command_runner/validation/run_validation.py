import tensorflow as tf

import models.autoencoder_models as ae_models
import models.nnet_models as nn_models
from validator.model_validator import ModelValidator

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'mlp', 'Model to validate.')
flags.DEFINE_string('method', 'split', 'Validation method.')
flags.DEFINE_integer('n_folds', 10, 'Number of cv folds.')
flags.DEFINE_float('test_size', 0.3, 'Percentage of data used for validation.')
flags.DEFINE_string('dataset_x', '', 'Path to the dataset file (.npy or .csv).')
flags.DEFINE_string('dataset_y', '', 'Path to the dataset outputs file (.npy or .csv).')


