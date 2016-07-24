import tensorflow as tf
import utils.utilities as utils
import command_runner.helper as cmd_helper

from models.autoencoder_models.stacked_denoising_autoencoder import StackedDenoisingAutoencoder


# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
cmd_helper.set_supervised_model_flags('sdae', flags)

# Denoising Autoencoder layers specific parameters
flags.DEFINE_string('layers', '512,256,', 'Comma-separated values for the layers in the SDAE.')
flags.DEFINE_string('dae_enc_act_func', 'sigmoid,', 'Activation function for the encoder. {}'.format(utils.valid_act_functions))
flags.DEFINE_string('dae_dec_act_func', 'none', 'Activation function for the decoder. {}'.format(utils.valid_act_functions))
flags.DEFINE_string('dae_cost_func', 'rmse', 'Cost function of each layer. {}'.format(utils.valid_unsupervised_cost_functions))
flags.DEFINE_string('dae_num_epochs', '30,', 'Number of training epochs of each layer.')
flags.DEFINE_string('dae_batch_size', '200,', 'Size of each training mini-batch of each layer.')
flags.DEFINE_string('dae_xavier_init', '1,', 'Value for the constant in xavier weights initialization.')
flags.DEFINE_string('dae_opt', 'adam,', 'Optmizer algorithm. {}'.format(utils.valid_optimization_functions))
flags.DEFINE_string('dae_learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('dae_momentum', '0.5,', 'Momentum parameter.')
flags.DEFINE_string('dae_corr_type', 'masking,', 'Input corruption type. ["masking", "gaussian"]')
flags.DEFINE_string('dae_corr_scale', '0.1,', 'Gaussian corruption scale.')
flags.DEFINE_string('dae_corr_keep_prob', '0.9,', 'Masking corruption keep probability.')
flags.DEFINE_string('dae_rho', '0.001,', 'Sparse autoencoder parameter rho.')
flags.DEFINE_string('dae_n_lambda', '3.0,', 'Sparse autoencoder parameter lambda.')
flags.DEFINE_string('dae_n_beta', '0.0001,', 'Sparse autoencoder parameter beta.')


# Global parameters
global_params = {
    'train_dataset': FLAGS.train_dataset,
    'train_labels':  FLAGS.train_labels,
    'test_dataset':  FLAGS.test_dataset,
    'test_labels':   FLAGS.test_labels,
    'valid_dataset': FLAGS.valid_dataset,
    'valid_labels':  FLAGS.valid_labels,
    'restore_model': FLAGS.restore_model,
}

# Get parameters
sdae_params = {
    'model_name':             FLAGS.model_name,
    'main_dir':               FLAGS.main_dir,
    'layers':                 utils.flag_to_list(FLAGS.layers, 'int'),
    'enc_act_func':           utils.flag_to_list(FLAGS.dae_enc_act_func, 'str'),
    'dec_act_func':           utils.flag_to_list(FLAGS.dae_dec_act_func, 'str'),
    'cost_func':              utils.flag_to_list(FLAGS.dae_cost_func, 'str'),
    'num_epochs':             utils.flag_to_list(FLAGS.dae_num_epochs, 'int'),
    'batch_size':             utils.flag_to_list(FLAGS.dae_batch_size, 'int'),
    'xavier_init':            utils.flag_to_list(FLAGS.dae_xavier_init, 'int'),
    'opt':                    utils.flag_to_list(FLAGS.dae_opt, 'str'),
    'learning_rate':          utils.flag_to_list(FLAGS.dae_learning_rate, 'float'),
    'momentum':               utils.flag_to_list(FLAGS.dae_momentum, 'float'),
    'corr_type':              utils.flag_to_list(FLAGS.dae_corr_type, 'str'),
    'corr_scale':             utils.flag_to_list(FLAGS.dae_corr_scale, 'float'),
    'corr_keep_prob':         utils.flag_to_list(FLAGS.dae_corr_keep_prob, 'float'),
    'rho':                    utils.flag_to_list(FLAGS.dae_rho, 'float'),
    'n_beta':                 utils.flag_to_list(FLAGS.dae_n_beta, 'float'),
    'n_lambda':               utils.flag_to_list(FLAGS.dae_n_lambda, 'float'),
    'hidden_dropout':         float(FLAGS.dropout),
    'finetune_cost_func':     FLAGS.cost_func,
    'finetune_act_func':      FLAGS.enc_act_func,
    'finetune_opt':           FLAGS.opt,
    'finetune_learning_rate': FLAGS.learning_rate,
    'finetune_momentum':      FLAGS.momentum,
    'finetune_num_epochs':    FLAGS.num_epochs,
    'finetune_batch_size':    FLAGS.batch_size,
    'seed':                   FLAGS.seed,
    'verbose':                FLAGS.verbose,
    'task':                   FLAGS.task
}


if __name__ == '__main__':

    # Create the SDAE object
    sdae = StackedDenoisingAutoencoder(**sdae_params)

    cmd_helper.run_supervised_model(sdae, global_params)