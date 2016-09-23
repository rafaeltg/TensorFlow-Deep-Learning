import tensorflow as tf

import pydl.utils.utilities as utils
from command_runner.cmd_flags import set_supervised_model_flags
from command_runner.cmd_model_run import run_supervised_model
from pydl.models.nnet_models.rnn import RNN

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

set_supervised_model_flags('lstm', flags)
flags.DEFINE_string('cell_type', 'simple', 'Recurrent layers type. ["lstm", "gru", "simple"]')
flags.DEFINE_string('layers', '50,50', 'String representing the architecture of the network.')
flags.DEFINE_boolean('stateful', True, 'Whether the recurrent network is stateful or not.')


# Global parameters
global_params = {
    'train_dataset':    FLAGS.train_dataset,
    'train_labels':     FLAGS.train_labels,
    'test_dataset':     FLAGS.test_dataset,
    'test_labels':      FLAGS.test_labels,
    'valid_dataset':    FLAGS.valid_dataset,
    'valid_labels':     FLAGS.valid_labels,
    'restore_model':    FLAGS.restore_model,
    'save_predictions': FLAGS.save_predictions
}

# Get parameters
rnn_params = {
    'model_name':    FLAGS.model_name,
    'main_dir':      FLAGS.main_dir,
    'cell_type':     FLAGS.cell_type,
    'layers':        utils.flag_to_list(FLAGS.layers, 'int'),
    'stateful':      FLAGS.stateful,
    'enc_act_func':  FLAGS.enc_act_func,
    'dec_act_func':  FLAGS.dec_act_func,
    'loss_func':     FLAGS.loss_func,
    'num_epochs':    FLAGS.num_epochs,
    'batch_size':    FLAGS.batch_size,
    'opt':           FLAGS.opt,
    'learning_rate': FLAGS.learning_rate,
    'momentum':      FLAGS.momentum,
    'dropout':       float(FLAGS.dropout),
    'verbose':       FLAGS.verbose,
    'seed':          FLAGS.seed
}


if __name__ == '__main__':

    run_supervised_model(RNN(**rnn_params), global_params)
