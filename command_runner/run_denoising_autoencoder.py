import tensorflow as tf

from command_runner.cmd_flags import set_unsupervised_model_flags
from command_runner.cmd_model_run import run_unsupervised_model
from pydl.models.autoencoder_models.denoising_autoencoder import DenoisingAutoencoder

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
set_unsupervised_model_flags('dae', flags)

# Denoising Autoencoder specific parameters
flags.DEFINE_integer('n_hidden', 32, 'Number of hidden units of the Denoising Autoencoder.')
flags.DEFINE_string('corr_type', 'gaussian', 'Type of input corruption. ["masking", "gaussian"]')
flags.DEFINE_float('corr_param', 0.3, '')

# Global parameters
global_params = {
    'train_dataset':     FLAGS.train_dataset,
    'test_dataset':      FLAGS.test_dataset,
    'valid_dataset':     FLAGS.valid_dataset,
    'restore_model':     FLAGS.restore_model,
    'save_encode_train': FLAGS.save_encode_train,
    'save_encode_valid': FLAGS.save_encode_valid,
    'save_encode_test':  FLAGS.save_encode_test,
}

# DAE parameters
dae_params = {
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
    'corr_type':     FLAGS.corr_type,
    'corr_param':    FLAGS.corr_param,
    'verbose':       FLAGS.verbose,
    'seed':          FLAGS.seed,
}


if __name__ == '__main__':

    # Create the Denoising Autoencoder object
    dae = DenoisingAutoencoder(**dae_params)

    run_unsupervised_model(dae, global_params)
