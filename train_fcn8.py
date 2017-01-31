#!/usr/bin/env python2

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
from models.fcn8 import buildFCN8
from metrics import jaccard, accuracy, crossentropy

_FLOATX = config.floatX
if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/Tmp/romerosa/itinf/models/camvid/fcn8_model.npz'
elif getuser() == 'jegousim':
    SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    LOADPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/rnncnn/fcn8_model.npz'
elif getuser() == 'michal':
    SAVEPATH = '/home/michal/Experiments/iter_inf/'
    LOADPATH = SAVEPATH
    WEIGHTS_PATH = '/home/michal/model_earlyjacc.npz'
elif getuser() == 'erraqaba':
    SAVEPATH = '/Tmp/erraqaba/iterative_inference/models/'
    LOADPATH = '/data/lisatmp4/erraqabi/iterative_inference/models/'
    WEIGHTS_PATH = LOADPATH
else:
    raise ValueError('Unknown user : {}'.format(getuser()))


def train(dataset, learn_step=0.005,
          weight_decay=1e-4, num_epochs=500,
          max_patience=100, data_augmentation={},
          savepath=None, loadpath=None,
          early_stop_class=None,
          batch_size=None,
          resume=False):

    #
    # Prepare load/save directories
    #
    exp_name = 'fcn8_' + 'data_aug' if bool(data_augmentation) else ''

    if savepath is None:
        raise ValueError('A saving directory must be specified')

    savepath = os.path.join(savepath, dataset, exp_name)
    loadpath = os.path.join(loadpath, dataset, exp_name)
    print savepath
    print loadpath

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    else:
        print('\033[93m The following folder already exists {}. '
              'It will be overwritten in a few seconds...\033[0m'.format(
                  savepath))

    print('Saving directory : ' + savepath)
    with open(os.path.join(savepath, "config.txt"), "w") as f:
        for key, value in locals().items():
            f.write('{} = {}\n'.format(key, value))

    #
    # Define symbolic variables
    #
    input_var = T.tensor4('input_var')
    target_var = T.ivector('target_var')

    #
    # Build dataset iterator
    #
    if batch_size is not None:
        bs = batch_size
    else:
        bs = [10, 1, 1]

    train_iter, val_iter, test_iter = \
        load_data(dataset, data_augmentation,
                  one_hot=False, batch_size=bs)

    n_batches_train = train_iter.nbatches
    n_batches_val = val_iter.nbatches
    n_batches_test = test_iter.nbatches if test_iter is not None else 0
    n_classes = train_iter.non_void_nclasses
    void_labels = train_iter.void_labels
    nb_in_channels = train_iter.data_shape[0]

    print "Batch. train: %d, val %d, test %d" % (n_batches_train, n_batches_val,
                                                 n_batches_test)
    print "Nb of classes: %d" % (n_classes)
    print "Nb. of input channels: %d" % (nb_in_channels)

    #
    # Build network
    #
    convmodel = buildFCN8(nb_in_channels, input_var, n_classes=n_classes,
                          void_labels=void_labels, trainable=True,
                          load_weights=resume, pascal=True, layer=['probs'])

    #
    # Define and compile theano functions
    #
    print "Defining and compiling training functions"
    prediction = lasagne.layers.get_output(convmodel)[0]
    loss = crossentropy(prediction, target_var, void_labels)

    if weight_decay > 0:
        weightsl2 = regularize_network_params(
            convmodel, lasagne.regularization.l2)
        loss += weight_decay * weightsl2

    params = lasagne.layers.get_all_params(convmodel, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learn_step)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print "Defining and compiling test functions"
    test_prediction = lasagne.layers.get_output(convmodel,
                                                deterministic=True)[0]
    test_loss = crossentropy(test_prediction, target_var, void_labels)
    test_acc = accuracy(test_prediction, target_var, void_labels)
    test_jacc = jaccard(test_prediction, target_var, n_classes)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc,
                                                       test_jacc])

    #
    # Train
    #
    err_train = []
    err_valid = []
    acc_valid = []
    jacc_valid = []
    patience = 0

    # Training main loop
    print "Start training"
    for epoch in range(num_epochs):
        # Single epoch training and validation
        start_time = time.time()
        cost_train_tot = 0

        # Train
        for i in range(n_batches_train):
            # Get minibatch
            X_train_batch, L_train_batch = train_iter.next()
            L_train_batch = np.reshape(L_train_batch,
                                       np.prod(L_train_batch.shape))

            # Training step
            cost_train = train_fn(X_train_batch, L_train_batch)
            out_str = "cost %f" % (cost_train)
            cost_train_tot += cost_train

        err_train += [cost_train_tot/n_batches_train]

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

        err_valid += [cost_val_tot/n_batches_val]
        acc_valid += [acc_val_tot/n_batches_val]
        jacc_perclass_valid = jacc_val_tot[0, :] / jacc_val_tot[1, :]
        if early_stop_class == None:
            jacc_valid += [np.mean(jacc_perclass_valid)]
        else:
            jacc_valid += [jacc_perclass_valid[early_stop_class]]


        out_str = "EPOCH %i: Avg epoch training cost train %f, cost val %f" +\
            ", acc val %f, jacc val %f took %f s"
        out_str = out_str % (epoch, err_train[epoch],
                             err_valid[epoch],
                             acc_valid[epoch],
                             jacc_valid[epoch],
                             time.time()-start_time)
        print out_str

        with open(os.path.join(savepath, "fcn8_output.log"), "a") as f:
            f.write(out_str + "\n")

        # Early stopping and saving stuff
        if epoch == 0:
            best_jacc_val = jacc_valid[epoch]
        elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
            best_jacc_val = jacc_valid[epoch]
            patience = 0
            np.savez(os.path.join(savepath, 'fcn8_model_best.npz'),
                     *lasagne.layers.get_all_param_values(convmodel))
            np.savez(os.path.join(savepath + "fcn8_errors_best.npz"),
                     err_valid, err_train, acc_valid,
                     jacc_valid)
        else:
            patience += 1
            np.savez(os.path.join(savepath, 'fcn8_model_last.npz'),
                     *lasagne.layers.get_all_param_values(convmodel))
            np.savez(os.path.join(savepath + "fcn8_errors_last.npz"),
                     err_valid, err_train, acc_valid,
                     jacc_valid)
        # Finish training if patience has expired or max nber of epochs
        # reached
        if patience == max_patience or epoch == num_epochs-1:
            if test_iter is not None:
                # Load best model weights
                with np.load(os.path.join(savepath, 'fcn8_model_best.npz')) as f:
                    param_values = [f['arr_%d' % i]
                                    for i in range(len(f.files))]
                nlayers = len(lasagne.layers.get_all_params(convmodel))
                lasagne.layers.set_all_param_values(convmodel,
                                                    param_values[:nlayers])
                # Test
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

                err_test = cost_test_tot/n_batches_test
                acc_test = acc_test_tot/n_batches_test
                jacc_test = np.mean(jacc_num_test_tot / jacc_denom_test_tot)

                out_str = "FINAL MODEL: err test % f, acc test %f, jacc test %f"
                out_str = out_str % (err_test,
                                     acc_test,
                                     jacc_test)
                print out_str
            if savepath != loadpath:
                print('Copying model and other training files to {}'.format(loadpath))
                copy_tree(savepath, loadpath)

            # End
            return


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-learning_rate',
                        default=0.0001,
                        help='Learning Rate')
    parser.add_argument('-penal_cst',
                        default=0.0,
                        help='regularization constant')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=750,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')
    parser.add_argument('-max_patience',
                        type=int,
                        default=100,
                        help='Max patience')
    parser.add_argument('-batch_size',
                        type=int,
                        default=[10, 1, 1],
                        help='Batch size [train, val, test]')
    parser.add_argument('-data_augmentation',
                        type=dict,
                        default={'crop_size': (224, 224), 'horizontal_flip': True, 'fill_mode':'constant'},
                        help='use data augmentation')
    parser.add_argument('-early_stop_class',
                        type=int,
                        default=None,
                        help='class to early stop on')
    args = parser.parse_args()

    train(args.dataset, float(args.learning_rate),
          float(args.penal_cst), int(args.num_epochs), int(args.max_patience),
          data_augmentation=args.data_augmentation, early_stop_class=args.early_stop_class,
          batch_size=args.batch_size, savepath=SAVEPATH, loadpath=LOADPATH)

if __name__ == "__main__":
    main()
