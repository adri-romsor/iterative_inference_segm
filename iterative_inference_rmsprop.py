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
# from lasagne.objectives import squared_error
from lasagne.nonlinearities import softmax

from data_loader import load_data
from metrics import accuracy, jaccard, squared_error
from models.DAE_h import buildDAE
from models.fcn8 import buildFCN8
from models.fcn8_dae import buildFCN8_DAE
from models.contextmod_dae import buildDAE_contextmod
from helpers import save_img
from models.model_helpers import softmax4D
from helpers import build_experiment_name

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


def inference(dataset, learn_step=0.005, num_iter=500,
              training_loss='squared_error', layer_h=['pool5'],
              n_filters=64, noise=0.1, conv_before_pool=1, additional_pool=0,
              dropout=0., skip=False, unpool_type='standard', from_gt=True,
              save_perstep=False, which_set='test', data_aug=False,
              savepath=None, loadpath=None, temperature=1.0,
              dae_kind='standard', test_from_0_255=False):
    #
    # Define symbolic variables
    #
    input_x_var = T.tensor4('input_x_var')
    input_h_var = []
    name = ''
    for l in layer_h:
        input_h_var += [T.tensor4()]
        name += ('_'+l)
    # y_hat_var = T.tensor4('pred_y_var')
    y_hat_var = theano.shared(np.zeros((10, 10, 10, 10), dtype=_FLOATX))
    y_hat_var_metrics = T.tensor4('y_hat_var_metrics')
    target_var_4D = T.itensor4('target_var_4D')

    #
    # Build dataset iterator
    #
    if which_set == 'train':
        test_iter, _, _ = load_data(dataset, train_crop_size=None,
                                    one_hot=True, batch_size=[10, 10, 10],
                                    return_0_255=test_from_0_255)
    elif which_set == 'valid':
        _, test_iter, _ = load_data(dataset, train_crop_size=None,
                                    one_hot=True, batch_size=[10, 10, 10],
                                    return_0_255=test_from_0_255)
    if which_set == 'test':
        _, _, test_iter = load_data(dataset, train_crop_size=None,
                                    one_hot=True, batch_size=[10, 10, 10],
                                    return_0_255=test_from_0_255)

    colors = test_iter.cmap
    n_batches_test = test_iter.nbatches
    n_classes = test_iter.non_void_nclasses
    void_labels = test_iter.void_labels
    nb_in_channels = test_iter.data_shape[0]
    void = n_classes if any(void_labels) else n_classes+1

    #
    # Prepare load/save directories
    #
    exp_name = build_experiment_name(dae_kind, layer_h, training_loss, from_gt,
                                     noise, data_aug, temperature, n_filters,
                                     conv_before_pool, additional_pool, skip,
                                     unpool_type, dropout)

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
    # Build networks
    #
    print 'Building networks'
    # Build FCN8  with pre-trained weights up to layer_h + prediction
    fcn = buildFCN8(nb_in_channels, input_var=input_x_var,
                    n_classes=n_classes,
                    void_labels=void_labels,
                    trainable=False, load_weights=True,
                    layer=layer_h+['probs_dimshuffle'],
                    path_weights=WEIGHTS_PATH+dataset+'/fcn8_model.npz',
                    temperature=temperature)

    # Build DAE with pre-trained weights
    print ' Building DAE network'
    if dae_kind == 'standard':
        dae = buildDAE(input_h_var, y_hat_var, n_classes, layer_h,
                       noise, n_filters, conv_before_pool, additional_pool,
                       dropout=dropout, trainable=True,
                       void_labels=void_labels, skip=skip,
                       unpool_type=unpool_type, load_weights=True,
                       path_weights=loadpath, model_name='dae_model.npz',
                       out_nonlin=softmax)
    elif dae_kind == 'fcn8':
        dae = buildFCN8_DAE(input_h_var, y_hat_var, n_classes,
                            n_classes, layer_h=layer_h, noise=noise,
                            path_weights=loadpath, model_name='dae_model.npz',
                            trainable=True, load_weights=True)

    elif dae_kind == 'contextmod':
        dae = buildDAE_contextmod(input_h_var, y_hat_var, n_classes,
                                  concat_layers=layer_h, noise=noise,
                                  path_weights=loadpath,
                                  model_name='dae_model.npz',
                                  trainable=True, load_weights=True,
                                  out_nonlin=softmax)
    else:
        raise ValueError('Unknown dae kind')

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
    target_var_dimshuffle = target_var_4D.dimshuffle((0, 2, 3, 1))
    sh2 = target_var_dimshuffle.shape
    target_var_2D = target_var_dimshuffle.reshape((T.prod(sh2[:3]), sh2[3]))

    # derivative of energy wrt input
    de = - (pred_dae - y_hat_var)

    # MSE loss
    loss=  squared_error(y_hat_var_metrics, target_var_4D, void)
    loss_fn = theano.function([y_hat_var_metrics, target_var_4D], loss)
    pred_dae_fn = theano.function(input_h_var, pred_dae)

    # metrics to evaluate iterative inference
    test_acc = accuracy(y_hat_2D, target_var_2D, void_labels, one_hot=True)
    test_jacc = jaccard(y_hat_2D, target_var_2D, n_classes, one_hot=True)

    # functions to compute metrics
    val_fn = theano.function([y_hat_var_metrics, target_var_4D],
                             [test_acc, test_jacc])

    # Softmax function
    softmax_fn = theano.function([y_hat_var_metrics],
                                 softmax4D(y_hat_var_metrics))

    #
    # Infer
    #
    print 'Start infering'
    acc_tot = 0
    acc_tot_old = 0
    jacc_tot = 0
    jacc_tot_old = 0
    acc_tot_dae = 0
    jacc_tot_dae = 0
    for i in range(1):  # (n_batches_test):
        info_str = "Batch %d out of %d" % (i, n_batches_test)
        print info_str

        # Get minibatch
        X_test_batch, L_test_batch = test_iter.next()

        # Compute fcn prediction y and h
        pred_test_batch = pred_fcn_fn(X_test_batch)
        Y_test_batch = pred_test_batch[-1]
        H_test_batch = pred_test_batch[:-1]
        y_hat_var.set_value(Y_test_batch)

        # Compute metrics before iterative inference
        acc_old, jacc_old = val_fn(Y_test_batch, L_test_batch)
        acc_tot_old += acc_old
        jacc_tot_old += jacc_old
        rec_loss = loss_fn(Y_test_batch, L_test_batch[:, :void, :, :])

        jacc_mean = np.mean(jacc_tot_old[0, :] / jacc_tot_old[1, :])
        print '>>>>> BEFORE: %f, %f, %f' % (rec_loss, acc_tot_old, jacc_mean)

        # Compute rec loss by using DAE in a standard way
        Y_test_batch_dae = pred_dae_fn(*(H_test_batch))
        Y_test_batch_dae = softmax_fn(Y_test_batch_dae)

        acc_dae, jacc_dae = val_fn(Y_test_batch_dae, L_test_batch)
        jacc_dae_mean = np.mean(jacc_dae[0, :] / jacc_dae[1, :])
        rec_loss_dae = loss_fn(Y_test_batch_dae, L_test_batch[:, :void, :, :])

        print '>>>>> Loss DAE: ' + str(rec_loss_dae)
        print '      Acc DAE: ' + str(acc_dae)
        print '      Jaccard DAE: ' + str(jacc_dae_mean)

        # Updates (grad, shared variable to update, learning_rate)
        updates = lasagne.updates.adam([de], [y_hat_var],
                                       learning_rate=learn_step)
        # function to compute de
        de_fn = theano.function(input_h_var, de, updates=updates)

        # Iterative inference
        for it in range(num_iter):
            grad = de_fn(*(H_test_batch))
            y_hat_var.set_value(softmax_fn(y_hat_var.get_value()))

            if save_perstep:
                # Save images
                save_img(np.copy(X_test_batch),
                         np.copy(L_test_batch).argmax(1),
                         np.copy(y_hat_var.get_value()),
                         np.copy(Y_test_batch),
                         savepath, n_classes,
                         'batch' + str(i) + '_' + 'step' + str(it),
                         void_labels, colors)

            norm = np.linalg.norm(grad, axis=1).mean()
            if norm < _EPSILON:
                break
            # print norm
            acc_iter, jacc_iter = val_fn(y_hat_var.get_value(), L_test_batch)
            rec_loss = loss_fn(y_hat_var.get_value(),
                               L_test_batch[:, :void, :, :])
            print rec_loss, acc_iter, np.mean(jacc_iter[0, :]/jacc_iter[1, :])

        # Compute metrics
        acc, jacc = val_fn(y_hat_var.get_value(), L_test_batch)
        acc_tot += acc
        jacc_tot += jacc

        jacc_perclass_old = jacc_tot_old[0, :]/jacc_tot_old[1, :]
        jacc_perclass = jacc_tot[0, :]/jacc_tot[1, :]

        info_str = "    fcn8 acc %f, iter acc %f, fcn8 jacc %f, iter jacc %f"
        info_str = info_str % (acc_tot_old,
                               acc_tot,
                               np.mean(jacc_perclass_old),
                               np.mean(jacc_perclass)
                               )
        print info_str

        print ">>> Per class jaccard:"
        labs = test_iter.mask_labels

        for i in range(len(labs)-len(void_labels)):
            class_str = '    ' + labs[i] + ' : old ->  %f, new %f'
            class_str = class_str % (jacc_perclass_old[i], jacc_perclass[i])
            print class_str

        if not save_perstep:
            # Save images
            save_img(np.copy(X_test_batch),
                     np.copy(L_test_batch).argmax(1),
                     np.copy(y_hat_var.get_value()),
                     np.copy(Y_test_batch),
                     savepath, n_classes,
                     'batch' + str(i), void_labels, colors)

    acc_test = acc_tot/n_batches_test
    jacc_test = np.mean(jacc_perclass)
    acc_test_old = acc_tot_old/n_batches_test
    jacc_test_old = np.mean(jacc_perclass_old)

    out_str = "TEST: acc  % f, jacc %f, acc old %f, jacc old %f"
    out_str = out_str % (acc_test, jacc_test,
                         acc_test_old, jacc_test_old)
    print out_str

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
    parser.add_argument('-training_loss',
                        type=str,
                        default=['squared_error', 'squared_error_h', 'crossentropy'],
                        help='Training loss')
    parser.add_argument('-layer_h',
                        type=list,
                        default=['pool3'],
                        help='All h to introduce to the DAE')
    parser.add_argument('-noise',
                        type=float,
                        default=0.1,
                        help='Noise of DAE input.')
    parser.add_argument('-n_filters',
                        type=int,
                        default=64,
                        help='Nb filters DAE (1st lay, increases pow 2')
    parser.add_argument('-conv_before_pool',
                        type=int,
                        default=1,
                        help='Conv. before pool in DAE.')
    parser.add_argument('-additional_pool',
                        type=int,
                        default=2,
                        help='Additional pool DAE')
    parser.add_argument('-dropout',
                        type=float,
                        default=0.5,
                        help='Additional pool DAE')
    parser.add_argument('-skip',
                        type=bool,
                        default=True,
                        help='Whether to skip connections in DAE')
    parser.add_argument('-unpool_type',
                        type=str,
                        default='trackind',
                        help='Unpooling type - standard or trackind')
    parser.add_argument('-from_gt',
                        type=bool,
                        default=False,
                        help='Whether to train from GT (true) or fcn' +
                        'output (False)')
    parser.add_argument('-save_perstep',
                        type=bool,
                        default=False,
                        help='Save new segmentations after each step update')
    parser.add_argument('-which_set',
                        type=str,
                        default='train',
                        help='Inference set')
    parser.add_argument('-data_aug',
                        type=bool,
                        default=True,
                        help='Whether to do data augmentation')
    parser.add_argument('-temperature',
                        type=float,
                        default=1.0,
                        help='Apply temperature')

    parser.add_argument('-dae_kind',
                        type=str,
                        default='fcn8',
                        help='What kind of AE archictecture to use')
    parser.add_argument('-test_from_0_255',
                        type=bool,
                        default=False,
                        help='Whether to train from images within 0-255 range')

    args = parser.parse_args()

    inference(args.dataset, float(args.step), int(args.num_iter),
              args.training_loss, args.layer_h, noise=args.noise,
              n_filters=args.n_filters, conv_before_pool=args.conv_before_pool,
              additional_pool=args.additional_pool, dropout=args.dropout,
              skip=args.skip, unpool_type=args.unpool_type,
              from_gt=args.from_gt, save_perstep=args.save_perstep,
              which_set=args.which_set, data_aug=args.data_aug,
              temperature=args.temperature, dae_kind=args.dae_kind,
              savepath=SAVEPATH, loadpath=LOADPATH,
              test_from_0_255=args.test_from_0_255)


if __name__ == "__main__":
    main()
