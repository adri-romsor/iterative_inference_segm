import os
import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree
import sys
import shutil

import numpy as np
import theano
import theano.tensor as T
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams
import lasagne
from lasagne.regularization import regularize_network_params

from data_loader import load_data
from models.DenseNet import buildDenseNet, summary
from metrics import jaccard, accuracy, crossentropy

_FLOATX = config.floatX
if getuser() == 'romerosa':
    savepath = '/Tmp/romerosa/itinf/models/'
    final_savepath = '/data/lisatmp4/romerosa/itinf/models/'
elif getuser() == 'jegousim':
    # savepath = '/Tmp/jegousim/iterative_inference/'
    savepath = '/data/lisatmp4/jegousim/iterative_inference/'
    final_savepath = '/data/lisatmp4/jegousim/iterative_inference/'
elif getuser() == 'michal':
    savepath = '/home/michal/Experiments/iter_inf/'
else:
    raise ValueError('Unknown user : {}'.format(getuser()))


def train(cf):
    # Save and print configuration
    print('-' * 75)
    print('Config\n')
    print('Local saving directory : ' + savepath)
    print('Final saving directory : ' + final_savepath)
    with open(os.path.join(savepath, "config.txt"), "w") as f:
        for key, value in cf.__dict__.items():
            if not key.startswith('__') & key.endswith('__'):
                f.write('{} = {}\n'.format(key, value))
                print('{} = {}'.format(key, value))

    # We also copy the model and the training scipt to reproduce exactly the experiments
    shutil.copy('train_densenet.py', os.path.join(savepath, 'train_densenet.py'))
    shutil.copy('models/DenseNet.py', os.path.join(savepath, 'DenseNet.py'))

    print('-' * 75)

    # Define symbolic variables
    input_var = T.tensor4('input_var')
    target_var = T.ivector('target_var')

    # Build dataset iterator
    print('Loading data')
    train_iter, val_iter, test_iter = load_data(cf.dataset,
                                                train_crop_size=cf.train_crop_size,
                                                batch_size=cf.batch_size,
                                                one_hot=False)

    n_batches_train = train_iter.get_n_batches()
    n_batches_val = val_iter.get_n_batches()
    n_batches_test = test_iter.get_n_batches()
    n_classes = cf.n_classes = train_iter.get_n_classes()
    void_labels = train_iter.get_void_labels()

    # Build model

    print('Building model and training functions')
    convmodel = buildDenseNet(
        cf.nb_in_channels,
        None,  # n_rows
        None,  # n_cols
        input_var,
        n_classes,
        cf.n_filters_first_conv,
        cf.filter_size,
        cf.n_blocks,
        cf.growth_rate,
        cf.n_conv_per_block,
        cf.dropout_p,
        cf.pad_mode,
        cf.pool_mode,
        cf.dilated_convolution_index,
        cf.upsampling_mode,
        cf.n_filters_deconvolution,
        cf.filter_size_deconvolution,
        cf.upsampling_block_mode,
        cf.trainable)

    summary(cf)

    # Compile training functions
    print('Compilation')

    prediction = lasagne.layers.get_output(convmodel)
    loss = cf.loss_function(prediction, target_var, void_labels)

    weightsl2 = regularize_network_params(convmodel, lasagne.regularization.l2)
    loss += cf.weight_decay * weightsl2

    params = lasagne.layers.get_all_params(convmodel, trainable=True)
    updates = cf.optimizer(loss, params, learning_rate=cf.learning_rate)

    start_time_compilation = time.time()
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

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

        batch_time = []
        for i in range(n_batches_train):
            # Get minibatch
            start_time_batch = time.time()
            X_train_batch, L_train_batch = train_iter.next()
            L_train_batch = np.reshape(L_train_batch, np.prod(L_train_batch.shape))

            # Training step
            cost_train = train_fn(X_train_batch, L_train_batch)
            cost_train_tot += cost_train

            # # Progression bar
            if epoch == 0:
                # We estimate the duration of a batch
                batch_time.append(time.time() - start_time_batch)
                mean_batch_time = np.mean(batch_time)

            # TODO : should not exceed 74 characters ...
            sys.stdout.write('\rEpoch {} : [{}%]. Remaining time = {} sec. Cost train = {:.4f}' \
                             .format(epoch, int(100. * (i + 1) / n_batches_train),
                                     int((n_batches_train - i + 1) * mean_batch_time),
                                     cost_train_tot / (i + 1)))
            sys.stdout.flush()

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

        out_str = \
            '\r\x1b[2Epoch {} took {} sec. Average cost train = {:.5f} | cost val = {:.5f} | ' \
            'acc val = {:.5f} | jacc_val = {:.5f} '.format(
                epoch, int(time.time() - start_time), err_train[-1], err_valid[-1], acc_valid[-1], jacc_valid[-1])

        # Early stopping and saving stuff
        if epoch == 0:
            best_jacc_val = jacc_valid[epoch]
        elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
            out_str += '(BEST)'
            best_jacc_val = jacc_valid[epoch]
            patience = 0
            np.savez(os.path.join(savepath, 'model.npz'), *lasagne.layers.get_all_param_values(convmodel))
            np.savez(os.path.join(savepath, 'errors.npz'), err_valid, err_train, acc_valid, jacc_valid)
        else:
            patience += 1

        print out_str

        with open(os.path.join(savepath, 'output.log'), 'a') as f:
            f.write(out_str + "\n")

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

    assert cf.exp_name is not None, 'Please provide a name for the experiment using -e name in the command line'

    cf.seed = 0
    np.random.seed(cf.seed)
    theano.tensor.shared_randomstreams.RandomStreams(cf.seed)
    # To make experiments reproductible, use deterministic convolution in CuDNN with THEANO_FLAGS

    # Training
    cf.dataset = 'camvid'
    cf.learning_rate = 0.0001
    cf.weight_decay = 0.
    cf.num_epochs = 500
    cf.max_patience = 100
    cf.loss_function = crossentropy
    cf.optimizer = lasagne.updates.rmsprop

    cf.nb_in_channels = 3
    cf.train_crop_size = (224, 224)
    cf.batch_size = [5, 5, 5]  # train / val / test

    # Architecture

    cf.n_filters_first_conv = 20
    cf.filter_size = 3
    cf.n_blocks = 5
    cf.growth_rate = 12
    cf.n_conv_per_block = [5, 7, 10, 10, 10, 10] + [5] * 5
    cf.dropout_p = 0.2
    cf.pad_mode = 'same'
    cf.pool_mode = 'average'
    cf.dilated_convolution_index = None
    cf.upsampling_mode = 'deconvolution'
    cf.n_filters_deconvolution = 'keep'
    cf.filter_size_deconvolution = 3
    cf.upsampling_block_mode = ('classic', [256, 128, 64, 32, 16])
    # cf.upsampling_block_mode = ('dense', 12)
    cf.trainable = True

    # Prepare save directories
    savepath = os.path.join(savepath, cf.dataset, cf.exp_name)  # local path
    final_savepath = os.path.join(final_savepath, cf.dataset, cf.exp_name)  # remote path

    for path in set([savepath, final_savepath]):  # in case savepath == final_savepath
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print('\033[93m The following folder already exists {}. '
                  'It will be overwritten in a few seconds...\033[0m'.format(path))

    try:
        train(cf)
    except KeyboardInterrupt:
        # In case of early stopping, transfer the local files
        if savepath != final_savepath:
            do_copy = raw_input('\033[93m KeyboardInterrupt \nDo you want to transfer files to {} ? ([y]/n) \033[0m'
                                .format(final_savepath))
            if do_copy in ['', 'y']:
                print('Copying model and other training files to {}'.format(final_savepath))
                copy_tree(savepath, final_savepath)