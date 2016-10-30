import pydl.utils.utilities as utils
import tensorflow as tf
from command_runner.cmd_flags import set_supervised_model_flags
from command_runner.cmd_model_run import run_supervised_model
from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder

# #################### #
#   Flags definition   #
# #################### #

flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
set_supervised_model_flags('sae', flags)

# Autoencoder layers specific parameters
flags.DEFINE_string('layers', '128,64,32,', 'Comma-separated values for the layers in the SAE.')
flags.DEFINE_string('ae_enc_act_func', 'relu,', 'Activation function for the encoder. {}'.format(utils.valid_act_functions))
flags.DEFINE_string('ae_dec_act_func', 'linear', 'Activation function for the decoder. {}'.format(utils.valid_act_functions))
flags.DEFINE_string('ae_l1_reg', '0.0,', 'L1 weight regularization penalty of each autoencoder.')
flags.DEFINE_string('ae_l2_reg', '0.0,', 'L2 weight regularization penalty of each autoencoder.')
flags.DEFINE_string('ae_loss_func', 'mse', 'Loss function of each layer. {}'.format(utils.valid_loss_functions))
flags.DEFINE_string('ae_num_epochs', '30,', 'Number of training epochs of each layer.')
flags.DEFINE_string('ae_batch_size', '200,', 'Size of each training mini-batch of each layer.')
flags.DEFINE_string('ae_opt', 'adam,', 'Optimizer algorithm. {}'.format(utils.valid_optimization_functions))
flags.DEFINE_string('ae_learning_rate', '0.01,', 'Initial learning rate.')
flags.DEFINE_string('ae_momentum', '0.5,', 'Momentum parameter.')


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
sae_params = {
    'model_name':             FLAGS.model_name,
    'main_dir':               FLAGS.main_dir,
    'layers':                 utils.flag_to_list(FLAGS.layers, 'int'),
    'enc_act_func':           utils.flag_to_list(FLAGS.ae_enc_act_func, 'str'),
    'dec_act_func':           utils.flag_to_list(FLAGS.ae_dec_act_func, 'str'),
    'l1_reg':                 utils.flag_to_list(FLAGS.ae_l1_reg, 'float'),
    'l2_reg':                 utils.flag_to_list(FLAGS.ae_l2_reg, 'float'),
    'loss_func':              utils.flag_to_list(FLAGS.ae_loss_func, 'str'),
    'num_epochs':             utils.flag_to_list(FLAGS.ae_num_epochs, 'int'),
    'batch_size':             utils.flag_to_list(FLAGS.ae_batch_size, 'int'),
    'opt':                    utils.flag_to_list(FLAGS.ae_opt, 'str'),
    'learning_rate':          utils.flag_to_list(FLAGS.ae_learning_rate, 'float'),
    'momentum':               utils.flag_to_list(FLAGS.ae_momentum, 'float'),
    'hidden_dropout':         float(FLAGS.dropout),
    'finetune_loss_func':     FLAGS.loss_func,
    'finetune_enc_act_func':  FLAGS.enc_act_func,
    'finetune_dec_act_func':  FLAGS.dec_act_func,
    'finetune_opt':           FLAGS.opt,
    'finetune_learning_rate': FLAGS.learning_rate,
    'finetune_momentum':      FLAGS.momentum,
    'finetune_num_epochs':    FLAGS.num_epochs,
    'finetune_batch_size':    FLAGS.batch_size,
    'seed':                   FLAGS.seed,
    'verbose':                FLAGS.verbose
}


if __name__ == '__main__':

    # Create the SAE object
    sae = StackedAutoencoder(**sae_params)

    run_supervised_model(sae, global_params)
