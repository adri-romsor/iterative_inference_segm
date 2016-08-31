import os
import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params

from data_loader import load_data
from models.DenseNet import buildDenseNet, summary
from metrics import jaccard, accuracy, crossentropy

_FLOATX = config.floatX
if getuser() == 'romerosa':
    LOCAL_SAVEPATH = '/Tmp/romerosa/itinf/models/'
    FINAL_SAVEPATH = '/data/lisatmp4/romerosa/itinf/models/'
elif getuser() == 'jegousim':
    LOCAL_SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    FINAL_SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
elif getuser() == 'michal':
    LOCAL_SAVEPATH = '/home/michal/Experiments/iter_inf/'
else:
    raise ValueError('Unknown user : {}'.format(getuser()))


def train(cf):
    # Prepare save directories
    savepath = os.path.join(LOCAL_SAVEPATH, cf.dataset, cf.exp_name)  # local path
    final_savepath = os.path.join(LOCAL_SAVEPATH, cf.dataset, cf.exp_name)  # remote path

    for path in set([savepath, final_savepath]):  # in case savepath == final_savepath
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('\033[93m The following folder already exists {}. '
                  'It will be overwritten in a few seconds...\033[0m'.format(path))

    # Save and print  configuration
    print('Saving directory : ' + savepath)
    print('-'*75)
    print('Config\n')
    with open(os.path.join(savepath, "config.txt"), "w") as f:
        for key, value in cf.__dict__.items():
            if not key.startswith('__') & key.endswith('__'):
                f.write('{} = {}\n'.format(key, value))
                print('{} = {}'.format(key, value))
    print('-'*75)

    # Define symbolic variables
    input_var = T.tensor4('input_var')
    target_var = T.ivector('target_var')

    # Build dataset iterator
    print('Loading data')
    train_iter, val_iter, test_iter = load_data(cf.dataset, train_crop_size=cf.train_crop_size, one_hot=False)

    n_batches_train = train_iter.get_n_batches()
    n_batches_val = val_iter.get_n_batches()
    n_batches_test = test_iter.get_n_batches()
    n_classes = cf.n_classes = train_iter.get_n_classes()
    void_labels = train_iter.get_void_labels()

    # Build model

    print('Building model and training functions')
    convmodel = buildDenseNet(
        cf.nb_in_channels,
        None, # n_rows
        None, # n_cols
        input_var,
        n_classes,
        cf.n_filters_first_conv,
        cf.filter_size,
        cf.n_blocks,
        cf.growth_rate_down,
        cf.growth_rate_up,
        cf.n_conv_per_block_down,
        cf.n_conv_per_block_up,
        cf.dropout_p,
        cf.pad_mode,
        cf.pool_mode,
        cf.dilated_convolution_index,
        cf.upsampling_mode,
        cf.deconvolution_mode,
        cf.upsampling_block_mode,
        cf.trainable)

    summary(cf)

    # Compile training functions

    prediction = lasagne.layers.get_output(convmodel)
    loss = cf.loss_function(prediction, target_var, void_labels)

    weightsl2 = regularize_network_params(convmodel, lasagne.regularization.l2)
    loss += cf.weight_decay * weightsl2

    params = lasagne.layers.get_all_params(convmodel, trainable=True)
    updates = cf.optimizer(loss, params, learning_rate=cf.learning_rate)

    start_time_compilation = time.time()
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print('-'*75)
    print('Compilation')
    print('Train compilation took {:.3f} seconds'.format(time.time() - start_time_compilation))

    # Compile test functions
    test_prediction = lasagne.layers.get_output(convmodel, deterministic=True)
    test_loss = cf.loss_function(test_prediction, target_var, void_labels)
    test_acc = accuracy(test_prediction, target_var, void_labels)
    test_jacc = jaccard(test_prediction, target_var, n_classes)

    start_time_compilation = time.time()
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_jacc])
    print('Validation compilation took {:.3f} seconds'.format(time.time() - start_time_compilation))

    # Parameters setup
    err_train = []
    err_valid = []
    acc_valid = []
    jacc_valid = []
    patience = 0

    # Training main loop
    print('-' * 30)
    print('Start training')
    print('-' * 30)

    for epoch in range(cf.num_epochs):
        # Single epoch training and validation
        start_time = time.time()
        cost_train_tot = 0

        # Train
        for i in range(n_batches_train):
            # Get minibatch
            X_train_batch, L_train_batch = train_iter.next()
            L_train_batch = np.reshape(L_train_batch, np.prod(L_train_batch.shape))

            # Training step
            cost_train = train_fn(X_train_batch, L_train_batch)
            out_str = "cost %f" % (cost_train)
            cost_train_tot += cost_train

        err_train += [cost_train_tot / n_batches_train]

        # Validation
        cost_val_tot = 0
        acc_val_tot = 0
        jacc_val_tot = np.zeros((2, n_classes))
        for i in range(n_batches_val):
            # Get minibatch
            X_val_batch, L_val_batch = val_iter.next()
            L_val_batch = np.reshape(L_val_batch,
                                     np.prod(L_val_batch.shape))

            # Validation step
            cost_val, acc_val, jacc_val = val_fn(X_val_batch, L_val_batch)

            acc_val_tot += acc_val
            cost_val_tot += cost_val
            jacc_val_tot += jacc_val

        err_valid += [cost_val_tot / n_batches_val]
        acc_valid += [acc_val_tot / n_batches_val]
        jacc_valid += [np.mean(jacc_val_tot[0, :] /
                               jacc_val_tot[1, :])]

        out_str = "EPOCH %i: Avg epoch training cost train %f, cost val %f" + \
                  ", acc val %f, jacc val %f took %f s"
        out_str = out_str % (epoch, err_train[epoch],
                             err_valid[epoch],
                             acc_valid[epoch],
                             jacc_valid[epoch],
                             time.time() - start_time)
        print out_str

        with open(os.path.join(savepath, "output.log", "a")) as f:
            f.write(out_str + "\n")

        # Early stopping and saving stuff
        if epoch == 0:
            best_jacc_val = jacc_valid[epoch]
        elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
            best_jacc_val = jacc_valid[epoch]
            patience = 0
            np.savez(os.path.join(savepath, 'model.npz'),
                     *lasagne.layers.get_all_param_values(convmodel))
            np.savez(os.path.join(savepath, 'errors.npz'),
                     err_valid, err_train, acc_valid,
                     jacc_valid)
        else:
            patience += 1

        # Finish training if patience has expired or max nber of epochs reached

        if patience == cf.max_patience or epoch == cf.num_epochs - 1:
            # Load best model weights
            with np.load(os.path.join(savepath, 'model.npz')) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            nlayers = len(lasagne.layers.get_all_params(convmodel))
            lasagne.layers.set_all_param_values(convmodel, param_values[:nlayers])

            # Test
            print('Training ends\nTest')
            cost_test_tot = 0
            acc_test_tot = 0
            jacc_num_test_tot = np.zeros((1, n_classes))
            jacc_denom_test_tot = np.zeros((1, n_classes))
            for i in range(n_batches_test):
                # Get minibatch
                X_test_batch, L_test_batch = test_iter.next()
                L_test_batch = np.reshape(L_test_batch,
                                          np.prod(L_test_batch.shape))

                # Test step
                cost_test, acc_test, jacc_test = \
                    val_fn(X_test_batch, L_test_batch)
                jacc_num_test, jacc_denom_test = jacc_test

                acc_test_tot += acc_test
                cost_test_tot += cost_test
                jacc_num_test_tot += jacc_num_test
                jacc_denom_test_tot += jacc_denom_test

            err_test = cost_test_tot / n_batches_test
            acc_test = acc_test_tot / n_batches_test
            jacc_test = np.mean(jacc_num_test_tot / jacc_denom_test_tot)

            out_str = "FINAL MODEL: err test % f, acc test %f, jacc test %f"
            out_str = out_str % (err_test,
                                 acc_test,
                                 jacc_test)
            print out_str

            # Copy files to final_savepath
            if savepath != final_savepath:
                print('Copying model and other training files to {}'.format(final_savepath))
                copy_tree(savepath, final_savepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DenseNet training')
    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of the experiment')
    cf = parser.parse_args()

    # Training
    cf.dataset = 'camvid'
    cf.learning_rate = 0.0001
    cf.weight_decay = 0.
    cf.num_epochs = 500
    cf.max_patience = 100
    cf.loss_function = crossentropy
    cf.optimizer = lasagne.updates.rmsprop

    cf.nb_in_channels = 3  # TODO : souldn't be here!
    cf.train_crop_size = (224, 224)

    # Architecture

    cf.n_filters_first_conv = 20
    cf.filter_size = 3
    cf.n_blocks = 5
    cf.growth_rate_down = 12
    cf.growth_rate_up = 12
    cf.n_conv_per_block_down = 5
    cf.n_conv_per_block_up = 5
    cf.dropout_p = 0.2
    cf.pad_mode = 'same'
    cf.pool_mode = 'average'
    cf.dilated_convolution_index = None
    cf.upsampling_mode = 'deconvolution'
    cf.deconvolution_mode = 'reduce'
    cf.upsampling_block_mode = 'dense'
    cf.trainable = True

    train(cf)
