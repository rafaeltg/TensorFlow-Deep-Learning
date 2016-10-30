import pydl.utils.utilities as utils
import tensorflow as tf
from command_runner.cmd_flags import set_unsupervised_model_flags
from command_runner.cmd_model_run import run_unsupervised_model
from pydl.models.autoencoder_models.deep_autoencoder import DeepAutoencoder

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
set_unsupervised_model_flags('deep_ae', flags)

# Deep Autoencoder specific parameters
flags.DEFINE_string('n_hidden', '256,128,64', 'Number of hidden units of the Deep Autoencoder.')


# Global parameters
global_params = {
    'train_dataset':     FLAGS.train_dataset,
    'test_dataset':      FLAGS.test_dataset,
    'valid_dataset':     FLAGS.valid_dataset,
    'save_encode_train': FLAGS.save_encode_train,
    'save_encode_valid': FLAGS.save_encode_valid,
    'save_encode_test':  FLAGS.save_encode_test,
}

# Autoencoder parameters
deep_ae_params = {
    'model_name':    FLAGS.model_name,
    'main_dir':      FLAGS.main_dir,
    'n_hidden':      utils.flag_to_list(FLAGS.n_hidden, 'int'),
    'enc_act_func':  FLAGS.enc_act_func,
    'dec_act_func':  FLAGS.dec_act_func,
    'l1_reg':        FLAGS.l1_reg,
    'l2_reg':        FLAGS.l2_reg,
    'loss_func':     FLAGS.loss_func,
    'num_epochs':    FLAGS.num_epochs,
    'batch_size':    FLAGS.batch_size,
    'opt':           FLAGS.opt,
    'learning_rate': FLAGS.learning_rate,
    'momentum':      FLAGS.momentum,
    'verbose':       FLAGS.verbose,
    'seed':          FLAGS.seed,
}


if __name__ == '__main__':

    run_unsupervised_model(DeepAutoencoder(**deep_ae_params), global_params)
