import tensorflow as tf

from command_runner.cmd_flags import set_unsupervised_model_flags
from command_runner.cmd_model_run import run_unsupervised_model
from pydl.models.autoencoder_models.variational_autoencoder import VariationalAutoencoder

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
set_unsupervised_model_flags('vae', flags)

# Variational Autoencoder specific parameters
flags.DEFINE_integer('n_hidden', 256, 'Number of hidden units of each intermediate layer.')
flags.DEFINE_integer('n_latent', 5, 'Number of hidden units in the latent layer.')


# Global parameters
global_params = {
    'train_dataset':     FLAGS.train_dataset,
    'test_dataset':      FLAGS.test_dataset,
    'valid_dataset':     FLAGS.valid_dataset,
    'save_encode_train': FLAGS.save_encode_train,
    'save_encode_valid': FLAGS.save_encode_valid,
    'save_encode_test':  FLAGS.save_encode_test,
}

# Variational Autoencoder parameters
vae_params = {
    'model_name':    FLAGS.model_name,
    'main_dir':      FLAGS.main_dir,
    'n_latent':      FLAGS.n_latent,
    'n_hidden':      FLAGS.n_hidden,
    'enc_act_func':  FLAGS.enc_act_func,
    'dec_act_func':  FLAGS.dec_act_func,
    'num_epochs':    FLAGS.num_epochs,
    'batch_size':    FLAGS.batch_size,
    'opt':           FLAGS.opt,
    'learning_rate': FLAGS.learning_rate,
    'momentum':      FLAGS.momentum,
    'verbose':       FLAGS.verbose,
    'seed':          FLAGS.seed,
}


if __name__ == '__main__':

    run_unsupervised_model(VariationalAutoencoder(**vae_params), global_params)
