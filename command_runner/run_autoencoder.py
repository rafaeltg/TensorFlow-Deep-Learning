import tensorflow as tf

from command_runner.cmd_flags import set_unsupervised_model_flags
from command_runner.cmd_model_run import run_unsupervised_model
from models.autoencoder_models.autoencoder import Autoencoder

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
set_unsupervised_model_flags('ae', flags)

# Autoencoder specific parameters
flags.DEFINE_integer('n_hidden', 32, 'Number of hidden units of the Autoencoder.')


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
ae_params = {
    'model_name':    FLAGS.model_name,
    'main_dir':      FLAGS.main_dir,
    'n_hidden':      FLAGS.n_hidden,
    'enc_act_func':  FLAGS.enc_act_func,
    'dec_act_func':  FLAGS.dec_act_func,
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

    # Create the Autoencoder object
    ae = Autoencoder(**ae_params)

    run_unsupervised_model(ae, global_params)
