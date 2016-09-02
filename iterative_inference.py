import argparse
import os
from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import theano
import theano.tensor as T
from theano import config

import lasagne

from data_loader import load_data
from metrics import accuracy, jaccard
from models.DAE_h import buildDAE
from models.fcn8 import buildFCN8
from helpers import save_img

_FLOATX = config.floatX

if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/Tmp/romerosa/itinf/models/camvid/fcn8_model.npz'
elif getuser() == 'jegousim':
    SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    LOADPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/rnncnn/fcn8_model.npz'
else:
    raise ValueError('Unknown user : {}'.format(getuser()))

_EPSILON = 1e-3


def inference(dataset, layer_name=None, learn_step=0.005, num_iter=500,
              num_filters=[256], skip=False, unpool_type='standard',
              filter_size=[3], savepath=None, loadpath=None, exp_name=None,
              training_loss='squared_error'):

    #
    # Define symbolic variables
    #
    input_x_var = T.tensor4('input_x_var')
    input_h_var = []
    name = ''
    for l in layer_name:
        input_h_var += [T.tensor4()]
        name += ('_'+l)
    y_hat_var = T.tensor4('pred_y_var')
    input_dae_mask_var = T.tensor4('input_dae_mask_var')
    target_var = T.ivector('target_var')

    #
    # Build dataset iterator
    #
    _, _, test_iter = load_data(dataset, train_crop_size=None, one_hot=True,
                                batch_size=[10, 10, 10])

    n_batches_test = test_iter.get_n_batches()
    n_classes = test_iter.get_n_classes()
    void_labels = test_iter.get_void_labels()

    #
    # Prepare load/save directories
    #
    if exp_name is None:
        exp_name = '_'.join(layer_name)
        exp_name += '_' + training_loss + ('_skip' if skip else '')

    if savepath is None:
        raise ValueError('A saving directory must be specified')

    savepath = os.path.join(savepath, dataset, exp_name, 'img_plots')
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
    # Build FCN8 with pre-trained weights (network to initialize
    # inference)
    fcn_y = buildFCN8(3, input_var=input_x_var,
                      n_classes=n_classes,
                      void_labels=void_labels,
                      trainable=False, load_weights=True)

    # Build FCN8  with pre-trained weights up to layer_name (that one will
    # be used as input to the DAE)
    fcn_h = buildFCN8(3, input_var=input_x_var,
                      n_classes=n_classes,
                      void_labels=void_labels,
                      trainable=False, load_weights=True,
                      layer=layer_name)

    # Build DAE with pre-trained weights
    dae = buildDAE(input_h_var, input_dae_mask_var,
                   n_classes, layer_h=layer_name, filter_size=num_filters,
                   kernel_size=filter_size, trainable=False, load_weights=True,
                   void_labels=void_labels, skip=skip, unpool_type=unpool_type,
                   model_name='dae_model.npz',
                   path_weights=loadpath)

    #
    # Define and compile theano functions
    #
    print "Defining and compiling theano functions"
    # Define required theano functions and compile them
    # predictions of fcn and dae
    pred_fcn_y = lasagne.layers.get_output(fcn_y, deterministic=True)[0]
    pred_fcn_h = lasagne.layers.get_output(fcn_h, deterministic=True)
    pred_dae = lasagne.layers.get_output(dae, deterministic=True)

    # function to compute output of fcn_y and fcn_h
    pred_fcn_y_fn = theano.function([input_x_var], pred_fcn_y)
    pred_fcn_h_fn = theano.function([input_x_var], pred_fcn_h)

    # Reshape iterative inference output to b,01c
    y_hat_dimshuffle = y_hat_var.dimshuffle((0, 2, 3, 1))
    sh = y_hat_dimshuffle.shape
    y_hat_2D = y_hat_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # derivative of energy wrt input
    de = - (pred_dae - pred_fcn_y)

    # function to compute de
    de_fn = theano.function(input_h_var+[input_dae_mask_var, input_x_var], de)

    # metrics to evaluate iterative inference
    test_acc = accuracy(y_hat_2D, target_var, void_labels)
    test_jacc = jaccard(y_hat_2D, target_var, n_classes)

    # functions to compute metrics
    val_fn = theano.function([y_hat_var, target_var],
                             [test_acc, test_jacc])

    #
    # Infer
    #
    print 'Start infering'
    acc_tot = 0
    acc_tot_old = 0
    jacc_tot = 0
    jacc_tot_old = 0
    for i in range(n_batches_test):
        info_str = "Batch %d out of %d" % (i, n_batches_test)
        print info_str

        # Get minibatch
        X_test_batch, L_test_batch = test_iter.next()
        L_test_target = L_test_batch.argmax(1)
        L_test_target = np.reshape(L_test_target,
                                   np.prod(L_test_target.shape))
        L_test_target = L_test_target.astype('int32')

        # Compute fcn prediction y and h
        Y_test_batch = pred_fcn_y_fn(X_test_batch)
        H_test_batch = pred_fcn_h_fn(X_test_batch)

        # Compute metrics before iterative inference
        acc_old, jacc_old = val_fn(Y_test_batch, L_test_target)
        acc_tot_old += acc_old
        jacc_tot_old += jacc_old
        Y_test_batch_old = Y_test_batch

        # Iterative inference
        for it in range(num_iter):
            grad = de_fn(*(H_test_batch+[Y_test_batch, X_test_batch]))

            Y_test_batch = Y_test_batch - learn_step * grad

            if np.linalg.norm(grad) < _EPSILON:
                break

        # Compute metrics
        acc, jacc = val_fn(Y_test_batch, L_test_target)

        acc_tot += acc
        jacc_tot += jacc

        info_str = "    old acc %f, new acc %f, old jacc %f, new jacc %f"
        info_str = info_str % (acc_tot_old,
                               acc_tot,
                               np.mean(jacc_tot_old[0, :]/jacc_tot_old[1, :]),
                               np.mean(jacc_tot[0, :] / jacc_tot[1, :]))
        print info_str

        # Save images
        save_img(X_test_batch, L_test_batch.argmax(1), Y_test_batch,
                 Y_test_batch_old, savepath, n_classes,
                 'batch' + str(i), void_labels)

    acc_test = acc_tot/n_batches_test
    jacc_test = np.mean(jacc_tot[0, :] / jacc_tot[1, :])
    acc_test_old = acc_tot_old/n_batches_test
    jacc_test_old = np.mean(jacc_tot_old[0, :] / jacc_tot_old[1, :])

    out_str = "TEST: acc  % f, jacc %f, acc old %f, jacc old %f"
    out_str = out_str % (acc_test, jacc_test,
                         acc_test_old, jacc_test_old)
    print out_str

    # Move segmentations
    if savepath != loadpath:
        print('Copying images to {}'.format(loadpath))
        copy_tree(savepath, os.path.join(loadpath, 'img_plots'))


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        type=str,
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-layer_name',
                        type=list,
                        default=['pool5'],
                        help='All h to introduce to the DAE.')
    parser.add_argument('-step',
                        type=float,
                        default=.001,
                        help='Step')
    parser.add_argument('--num_iter',
                        '-nit',
                        type=int,
                        default=500,
                        help='Max number of iterations.')
    parser.add_argument('-num_filters',
                        type=list,
                        default=[2048],
                        help='All h to introduce to the DAE.')
    parser.add_argument('-skip',
                        type=bool,
                        default=True,
                        help='Whether to skip connections in the DAE.')
    parser.add_argument('-unpool_type',
                        type=str,
                        default='standard',
                        help='Unpool type - standard or trackind.')
    parser.add_argument('-training_loss',
                        type=str,
                        default='squared_error',
                        help='Training loss')
    parser.add_argument('--savepath',
                        '-sp',
                        type=str,
                        default=SAVEPATH,
                        help='Path to save images')
    parser.add_argument('--loadpath',
                        '-lp',
                        type=str,
                        default=LOADPATH,
                        help='Path to save images')

    args = parser.parse_args()

    inference(args.dataset, args.layer_name, float(args.step),
              int(args.num_iter), args.num_filters, args.skip,
              args.unpool_type, savepath=args.savepath,
              loadpath=args.loadpath, training_loss=args.training_loss)

if __name__ == "__main__":
    main()
