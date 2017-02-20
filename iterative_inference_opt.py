#!/usr/bin/env python2

import argparse
import os
from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import theano
import theano.tensor as T
from theano import config

import lasagne
from lasagne.nonlinearities import softmax
from lasagne.updates import sgd, adam, rmsprop

from data_loader import load_data
from metrics import accuracy, jaccard, squared_error
from models.DAE_h import buildDAE
from models.fcn8 import buildFCN8
from models.fcn8_dae import buildFCN8_DAE
from models.contextmod_dae import buildDAE_contextmod
from helpers import save_img
from models.model_helpers import softmax4D
from helpers import build_experiment_name, print_results

_FLOATX = config.floatX
_EPSILON = 10e-8

if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/itinf/models/'
elif getuser() == 'jegousim':
    SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    LOADPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/rnncnn/fcn8_model.npz'
else:
    raise ValueError('Unknown user : {}'.format(getuser()))

_EPSILON = 1e-3


def inference(dataset, learn_step=0.005, num_iter=500, optimizer=sgd,
              dae_dict_updates= {}, training_dict={}, data_augmentation=False,
              save_perstep=False, which_set='test',
              savepath=None, loadpath=None, test_from_0_255=False):

    #
    # Update DAE parameters
    #
    dae_dict = {'kind': 'fcn8',
                'dropout': 0.0,
                'skip': True,
                'unpool_type':'standard',
                'n_filters': 64,
                'conv_before_pool': 1,
                'additional_pool': 0,
                'concat_h': ['input'],
                'noise': 0.0,
                'from_gt': True,
                'temperature': 1.0}

    dae_dict.update(dae_dict_updates)

    #
    # Prepare load/save directories
    #
    exp_name = build_experiment_name(data_aug=data_augmentation,
                                     **dict(dae_dict.items() + training_dict.items()))
    if savepath is None:
        raise ValueError('A saving directory must be specified')

    savepath = os.path.join(savepath, dataset, exp_name, 'img_plots',
                            which_set)
    loadpath = os.path.join(loadpath, dataset, exp_name)
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
    input_x_var = T.tensor4('input_x_var')  # tensor for input image batch
    input_concat_h_vars = [T.tensor4()] * len(dae_dict['concat_h'])  # tensor for hidden repr batch (input dae)
    target_var = T.tensor4('target_var')  # tensor for target batch
    y_hat_var = theano.shared(np.zeros((10, 10, 10, 10), dtype=_FLOATX))
    y_hat_var_metrics = T.tensor4('y_hat_var_metrics')

    #
    # Build dataset iterator
    #
    data_iter = load_data(dataset, {}, one_hot=True, batch_size=[10, 10, 10],
                          return_0_255=test_from_0_255, which_set=which_set)

    colors = data_iter.cmap
    n_batches_test = data_iter.nbatches
    n_classes = data_iter.non_void_nclasses
    void_labels = data_iter.void_labels
    nb_in_channels = data_iter.data_shape[0]
    void = n_classes if any(void_labels) else n_classes+1

    #
    # Build networks
    #

    # Build DAE with pre-trained weights
    print ' Building DAE network'
    if dae_dict['kind'] == 'standard':
        dae = buildDAE(input_concat_h_vars, y_hat_var, n_classes, trainable=True,
                       void_labels=void_labels, load_weights=True,
                       path_weights=loadpath, model_name='dae_model_best.npz',
                       out_nonlin=softmax, concat_h=dae_dict['concat_h'],
                       noise=dae_dict['noise'], n_filters=dae_dict['n_filters'],
                       conv_before_pool=dae_dict['conv_before_pool'],
                       additional_pool=dae_dict['additional_pool'],
                       dropout=dae_dict['dropout'], skip=dae_dict['skip'],
                       unpool_type=dae_dict['unpool_type'])
    elif dae_dict['kind'] == 'fcn8':
        dae = buildFCN8_DAE(input_concat_h_vars, y_hat_var, n_classes,
                            nb_in_channels=n_classes, path_weights=loadpath,
                            model_name='dae_model_best.npz', trainable=True,
                            load_weights=True, pretrained=True, pascal=False,
                            concat_h=dae_dict['concat_h'], noise=dae_dict['noise'])
    elif dae_dict['kind'] == 'contextmod':
        dae = buildDAE_contextmod(input_concat_h_vars, y_hat_var, n_classes,
                                  path_weights=loadpath,
                                  model_name='dae_model_best.npz',
                                  trainable=True, load_weights=True,
                                  out_nonlin=softmax, noise=dae_dict['noise'],
                                  concat_h=dae_dict['concat_h'])
    else:
        raise ValueError('Unknown dae kind')

    # Build FCN8  with pre-trained weights up to layer_h + prediction
    print ' Building FCN network'
    dae_dict['concat_h'] += ['probs_dimshuffle']
    fcn = buildFCN8(nb_in_channels, input_var=input_x_var,
                    n_classes=n_classes, void_labels=void_labels,
                    path_weights=WEIGHTS_PATH+dataset+'/new_fcn8_model_best.npz',
                    trainable=False, load_weights=True,
                    layer=dae_dict['concat_h'],temperature=dae_dict['temperature'])

    #
    # Define and compile theano functions
    #
    print "Defining and compiling theano functions"
    # Define required theano functions and compile them
    # predictions of fcn and dae
    pred_fcn = lasagne.layers.get_output(fcn, deterministic=True)
    pred_dae = lasagne.layers.get_output(dae, deterministic=True)

    # function to compute outputs of fcn
    pred_fcn_fn = theano.function([input_x_var], pred_fcn)

    # Reshape iterative inference output to b01,c
    y_hat_dimshuffle = y_hat_var_metrics.dimshuffle((0, 2, 3, 1))
    sh = y_hat_dimshuffle.shape
    y_hat_2D = y_hat_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # Reshape iterative inference output to b01,c
    target_var_dimshuffle = target_var.dimshuffle((0, 2, 3, 1))
    sh2 = target_var_dimshuffle.shape
    target_var_2D = target_var_dimshuffle.reshape((T.prod(sh2[:3]), sh2[3]))

    # derivative of energy wrt input
    de = - (pred_dae - y_hat_var)

    # metrics to evaluate iterative inference
    test_loss =  squared_error(y_hat_var_metrics, target_var, void)
    test_acc = accuracy(y_hat_2D, target_var_2D, void_labels, one_hot=True)
    test_jacc = jaccard(y_hat_2D, target_var_2D, n_classes, one_hot=True)
    pred_dae_fn = theano.function(input_concat_h_vars, pred_dae)

    # functions to compute metrics
    val_fn = theano.function([y_hat_var_metrics, target_var],
                             [test_acc, test_jacc, test_loss])

    # Softmax function
    # softmax_fn = theano.function([y_hat_var_metrics],
    #                              softmax4D(y_hat_var_metrics))

    # Clip function
    clip_fn = theano.function([y_hat_var_metrics],
                              T.clip(y_hat_var_metrics, 0.0, 1.0))

    #
    # Infer
    #
    print 'Start infering'
    rec_tot = 0
    rec_tot_fcn = 0
    rec_tot_dae = 0
    acc_tot = 0
    acc_tot_fcn = 0
    acc_tot_dae = 0
    jacc_tot = 0
    jacc_tot_fcn = 0
    jacc_tot_dae = 0
    for i in range(n_batches_test):
        info_str = "Batch %d out of %d" % (i, n_batches_test)
        print info_str

        # Get minibatch
        X_test_batch, L_test_batch = data_iter.next()
        L_test_batch = L_test_batch.astype(_FLOATX)

        # Compute fcn prediction y and h
        pred_test_batch = pred_fcn_fn(X_test_batch)
        Y_test_batch = pred_test_batch[-1]
        H_test_batch = pred_test_batch[:-1]
        y_hat_var.set_value(Y_test_batch)

        # Compute metrics before iterative inference
        acc_fcn, jacc_fcn, rec_fcn = val_fn(Y_test_batch, L_test_batch)
        acc_tot_fcn += acc_fcn
        jacc_tot_fcn += jacc_fcn
        rec_tot_fcn += rec_fcn
        print_results('>>>>> FCN:', rec_tot_fcn, acc_tot_fcn, jacc_tot_fcn, i+1)

        # Compute rec loss by using DAE in a standard way
        Y_test_batch_dae = pred_dae_fn(*(H_test_batch))

        acc_dae, jacc_dae, rec_dae = val_fn(Y_test_batch_dae, L_test_batch)
        acc_tot_dae += acc_dae
        jacc_tot_dae += jacc_dae
        rec_tot_dae += rec_dae
        print_results('>>>>> FCN+DAE:', rec_tot_dae, acc_tot_dae, jacc_tot_dae, i+1)

        # Updates (grad, shared variable to update, learning_rate)
        updates = optimizer([de], [y_hat_var], learning_rate=learn_step)

        # function to compute de
        de_fn = theano.function(input_concat_h_vars, de, updates=updates)

        # Iterative inference
        for it in range(num_iter):
            grad = de_fn(*(H_test_batch))
            y_hat_var.set_value(clip_fn(y_hat_var.get_value()))

            if save_perstep:
                # Save images
                save_img(np.copy(X_test_batch),
                         np.copy(L_test_batch),
                         np.copy(y_hat_var.get_value()),
                         np.copy(Y_test_batch),
                         savepath,
                         'batch' + str(i) + '_' + 'step' + str(it),
                         void_labels, colors)

            norm = np.linalg.norm(grad, axis=1).mean()
            if norm < _EPSILON:
                break

            acc_iter, jacc_iter, rec_iter = val_fn(y_hat_var.get_value(), L_test_batch)
            print rec_iter, acc_iter, np.mean(jacc_iter[0, :]/jacc_iter[1, :])

        # Compute metrics
        acc, jacc, rec = val_fn(y_hat_var.get_value(), L_test_batch)
        acc_tot += acc
        jacc_tot += jacc
        rec_tot += rec
        print_results('>>>>> ITERATIVE INFERENCE:', rec_tot, acc_tot, jacc_tot, i+1)

        if not save_perstep:
            # Save images
            save_img(np.copy(X_test_batch),
                     np.copy(L_test_batch).argmax(1),
                     np.copy(y_hat_var.get_value()),
                     np.copy(Y_test_batch),
                     savepath, 'batch' + str(i),
                     void_labels, colors)

    # Print summary of how things went
    print('-------------------------------------------------------------------')
    print('------------------------------SUMMARY------------------------------')
    print('-------------------------------------------------------------------')
    print_results('>>>>> FCN:', rec_tot_fcn, acc_tot_fcn, jacc_tot_fcn, i+1)
    print_results('>>>>> FCN+DAE:', rec_tot_dae, acc_tot_dae, jacc_tot_dae, i+1)
    print_results('>>>>> ITERTIVE INFERENCE:', rec_tot, acc_tot, jacc_tot, i+1)

    # Compute per class jaccard
    jacc_perclass_fcn = jacc_tot_fcn[0, :]/jacc_tot_fcn[1, :]
    jacc_perclass = jacc_tot[0, :]/jacc_tot[1, :]

    print ">>>>> Per class jaccard:"
    labs = data_iter.mask_labels

    for i in range(len(labs)-len(void_labels)):
        class_str = '    ' + labs[i] + ' : fcn ->  %f, ii ->  %f'
        class_str = class_str % (jacc_perclass_fcn[i], jacc_perclass[i])
        print class_str

    # Move segmentations
    if savepath != loadpath:
        print('Copying images to {}'.format(loadpath))
        copy_tree(savepath, os.path.join(loadpath, 'img_plots', which_set))


def main():
    parser = argparse.ArgumentParser(description='Iterative inference.')

    parser.add_argument('-dataset',
                        type=str,
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-step',
                        type=float,
                        default=.05,
                        help='step')
    parser.add_argument('--num_iter',
                        '-ne',
                        type=int,
                        default=10,
                        help='Max number of iterations')
    parser.add_argument('-optimizer',
                        default=sgd,
                        help='Optimizer (sgd, rmsprop or adam)')
    parser.add_argument('-save_perstep',
                        type=bool,
                        default=False,
                        help='Save new segmentations after each step update')
    parser.add_argument('-which_set',
                        type=str,
                        default='val',
                        help='Inference set')
    parser.add_argument('-dae_dict',
                        type=dict,
                        default={'kind': 'fcn8', 'dropout': 0.5, 'skip': True,
                                  'unpool_type': 'standard', 'noise': 0.0,
                                  'concat_h': ['pool4'], 'from_gt': False,
                                  'n_filters': 64, 'conv_before_pool': 1,
                                  'additional_pool': 2, 'temperature': 1.0},
                        help='DAE kind and parameters')
    parser.add_argument('-training_dict',
                        type=dict,
                        default={'training_loss': ['crossentropy'],
                                 'learning_rate': 0.0001, 'lr_anneal': 0.99,
                                 'weight_decay':0.0001, 'optimizer': 'rmsprop'},
                        help='Training parameters')
    parser.add_argument('-data_augmentation',
                        type=bool,
                        default=True,
                        help='Whether we performed any kind of data augmentation during training.')
    parser.add_argument('-test_from_0_255',
                        type=bool,
                        default=True,
                        help='Whether to train from images within 0-255 range')

    args = parser.parse_args()

    inference(args.dataset, float(args.step), int(args.num_iter), args.optimizer,
              save_perstep=args.save_perstep, which_set=args.which_set,
              savepath=SAVEPATH, loadpath=LOADPATH,
              test_from_0_255=args.test_from_0_255,
              dae_dict_updates=args.dae_dict, data_augmentation=args.data_augmentation,
              training_dict=args.training_dict)


if __name__ == "__main__":
    main()
