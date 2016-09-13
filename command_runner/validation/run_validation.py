import tensorflow as tf
import numpy as np
import models.autoencoder_models as ae_models
import models.nnet_models as nn_models
from utils import datasets
from validator.model_validator import ModelValidator

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('method', 'split', 'Validation method.')
flags.DEFINE_integer('n_folds', 10, 'Number of cv folds.')
flags.DEFINE_float('test_size', 0.3, 'Percentage of data used for validation.')
flags.DEFINE_string('dataset_x', '', 'Path to the dataset file (.npy or .csv).')
flags.DEFINE_string('dataset_y', '', 'Path to the dataset outputs file (.npy or .csv).')
flags.DEFINE_string('metrics', 'mse,mae,rmse', '')


params = {
  'method':    FLAGS.method,
  'k':         FLAGS.n_folds,
  'test_size': FLAGS.test_size,
  'dataset_x': FLAGS.dataset_x,
  'dataset_y': FLAGS.dataset_y
  'metrics':   utils.flag_to_list(FLAGS.metrics, 'str'),
}


func run_validation(model, **kwargs) {
  
  validador = ModelValidator(method=kwargs.get('method'),
                             k=kwarg.get('k'),
                             test_size=kwargs.get('test_size'))
  
  res = validator.run(model=model,
                      x=kwargs.get('dataset_x'),
                      y=kwargs.get('dataset_y'),
                      metrics=kwargs.get('metrics'))
  
  print('----------------------------------')
  print('        VALIDATION RESULTS        ')
  
  print('> Test score = %f (std dev = %f)' % (np.mean(res['scores']), np.std(res['scores'])))
  
  for m, v in enumerate(res['metrics']):
    print('> %s = %f (std dev = %f)' % (m, np.mean(v), np.std(v)))
}


if __name__ == '__main__':
  
  model = nn_models.MLP() # default parameters
  
  dataset = datasets.load_dataset(params['dataset_x'], params['dataset_y'])
  
  run_validation(model, dataset.data, dataset.target, **params)
