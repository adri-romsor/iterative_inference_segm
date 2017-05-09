# To run the code install pydensecrf

import argparse
import os
import numpy as np

import theano
import theano.tensor as T
from theano import config
from getpass import getuser
from distutils.dir_util import copy_tree

import lasagne

from data_loader import load_data
from metrics import accuracy, jaccard
from models.DAE_h import buildDAE
from models.fcn8_void import buildFCN8
from models.FCDenseNet import build_fcdensenet
from helpers import save_img

import sys
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral,\
                             create_pairwise_gaussian, unary_from_softmax

_FLOATX = config.floatX

if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/itinf/models/'
elif getuser() == 'jegousim':
    SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    LOADPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/rnncnn/fcn8_model.npz'
elif getuser() == 'erraqaba':
    SAVEPATH = '/Tmp/erraqaba/iterative_inference/models/'
    LOADPATH = '/data/lisatmp4/erraqabi/iterative_inference/models/'
    WEIGHTS_PATH = LOADPATH
else:
    raise ValueError('Unknown user : {}'.format(getuser()))


def inference(dataset, segm_net, which_set='val', num_iter=5, Bilateral=True,
              savepath=None, loadpath=None, test_from_0_255=False):

    #
    # Define symbolic variables
    #
    input_x_var = T.tensor4('input_x_var')
    y_hat_var = T.tensor4('pred_y_var')
    target_var = T.tensor4('target_var')

    #
    # Build dataset iterator
    #
    data_iter = load_data(dataset, {}, one_hot=True, batch_size=[10, 10, 10],
                          return_0_255=test_from_0_255, which_set=which_set)

    colors = data_iter.cmap
    n_batches_test = data_iter.nbatches
    n_classes = data_iter.non_void_nclasses
    void_labels = data_iter.void_labels

    #
    # Prepare saving directory
    #
    savepath = os.path.join(savepath, dataset, segm_net, 'img_plots', which_set)
    loadpath = os.path.join(loadpath, dataset, segm_net, 'img_plots', which_set)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    #
    # Build network
    #
    print 'Building segmentation network'
    if segm_net == 'fcn8':
        fcn = buildFCN8(3, input_var=input_x_var,
                        n_classes=n_classes, void_labels=void_labels,
                        path_weights=WEIGHTS_PATH+dataset+'/fcn8_model.npz',
                        trainable=False, load_weights=True,
                        layer=['probs_dimshuffle'])
        padding = 100
    elif segm_net == 'densenet':
        fcn  = build_fcdensenet(input_x_var, nb_in_channels=3,
                                n_classes=n_classes, layer=[])
        padding = 0
    elif segm_net == 'fcn_fcresnet':
        raise NotImplementedError
    else:
        raise ValueError

    #
    # Define and compile theano functions
    #
    print "Defining and compiling theano functions"

    # predictions of fcn
    pred_fcn = lasagne.layers.get_output(fcn, deterministic=True, batch_norm_use_averages=False)[0]

    # function to compute output of fcn
    pred_fcn_fn = theano.function([input_x_var], pred_fcn)

    # reshape fcn output to b,01c
    y_hat_dimshuffle = y_hat_var.dimshuffle((0, 2, 3, 1))
    sh = y_hat_dimshuffle.shape
    y_hat_2D = y_hat_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # reshape target to b01,c
    target_var_dimshuffle = target_var.dimshuffle((0, 2, 3, 1))
    sh2 = target_var_dimshuffle.shape
    target_var_2D = target_var_dimshuffle.reshape((T.prod(sh2[:3]), sh2[3]))

    # metrics to evaluate iterative inference
    test_acc = accuracy(y_hat_2D, target_var_2D, void_labels, one_hot=True)
    test_jacc = jaccard(y_hat_2D, target_var_2D, n_classes, one_hot=True)

    # functions to compute metrics
    val_fn = theano.function([y_hat_var, target_var], [test_acc, test_jacc])

    #
    # Infer
    #
    print 'Start infering'
    acc_tot_crf = 0
    acc_tot_fcn = 0
    jacc_tot_crf = 0
    jacc_tot_fcn = 0
    for i in range(n_batches_test):
        info_str = "Batch %d out of %d" % (i+1, n_batches_test)
        print info_str

        # Get minibatch
        X_test_batch, L_test_batch = data_iter.next()
        L_test_batch = L_test_batch.astype(_FLOATX)

        # Compute fcn prediction
        Y_test_batch = pred_fcn_fn(X_test_batch)

        # Compute metrics before CRF
        acc_fcn, jacc_fcn = val_fn(Y_test_batch, L_test_batch)
        acc_tot_fcn += acc_fcn
        jacc_tot_fcn += jacc_fcn
        Y_test_batch_fcn = Y_test_batch
        Y_test_batch_crf = []

        for im in range(X_test_batch.shape[0]):
            # CRF
            d = dcrf.DenseCRF2D(Y_test_batch.shape[3], Y_test_batch.shape[2],
                                n_classes)
            sm = Y_test_batch[im, 0:n_classes, :, :]
            sm = sm.reshape((n_classes, -1))
            img = X_test_batch[im]
            img = np.transpose(img, (1, 2, 0))
            img = (255 * img).astype('uint8')
            img2 = np.asarray(img, order='C')

            # set unary potentials (neg log probability)
            U = unary_from_softmax(sm)
            d.setUnaryEnergy(U)

            # set pairwise potentials

            # This adds the color-independent term, features are the
            # locations only. Smoothness kernel.
            # sxy: gaussian x, y std
            # compat: ways to weight contributions, a number for potts compatibility,
            #     vector for diagonal compatibility, an array for matrix compatibility
            # kernel: kernel used, CONST_KERNEL, FULL_KERNEL, DIAG_KERNEL
            # normalization: NORMALIZE_AFTER, NORMALIZE_BEFORE,
            #     NO_NORMALIZAITION, NORMALIZE_SYMMETRIC
            d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                                  normalization=dcrf.NORMALIZE_SYMMETRIC)

            if Bilateral:
                # Appearance kernel. This adds the color-dependent term, i.e. features
                # are (x,y,r,g,b).
                # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
                # to set sxy and srgb perform grid search on validation set
                d.addPairwiseBilateral(sxy=(3, 3), srgb=(13, 13, 13),
                                       rgbim=img2, compat=10, kernel=dcrf.DIAG_KERNEL,
                                       normalization=dcrf.NORMALIZE_SYMMETRIC)

            # inference
            Q = d.inference(num_iter)
            Q = np.reshape(Q, (n_classes, Y_test_batch.shape[2], Y_test_batch.shape[3]))
            Y_test_batch_crf += [np.expand_dims(Q, axis=0)]

            # Compute metrics after CRF
            acc_crf, jacc_crf = val_fn(Y_test_batch_crf[im], L_test_batch[np.newaxis, im, :, :])
            acc_tot_crf += acc_crf
            jacc_tot_crf += jacc_crf

        # Save images
        Y_test_batch = np.concatenate(Y_test_batch_crf, axis=0)
        save_img(X_test_batch.astype(_FLOATX), L_test_batch, Y_test_batch,
                 Y_test_batch_fcn, savepath, 'batch' + str(i), void_labels, colors)

    acc_test_crf = acc_tot_crf/n_batches_test
    jacc_test_crf = np.mean(jacc_tot_crf[0, :] / jacc_tot_crf[1, :])
    acc_test_fcn = acc_tot_fcn/n_batches_test
    jacc_test_fcn = np.mean(jacc_tot_fcn[0, :] / jacc_tot_fcn[1, :])

    out_str = "TEST: acc crf  % f, jacc crf %f, acc fcn %f, jacc fcn %f"
    out_str = out_str % (acc_test_crf, jacc_test_crf,
                         acc_test_fcn, jacc_test_fcn)
    print out_str

    # Move segmentations
    if savepath != loadpath:
        print('Copying images to {}'.format(loadpath))
        copy_tree(savepath, loadpath)


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        type=str,
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-segmentation_net',
                        type=str,
                        default='fcn8',
                        help='Segmentation network.')
    parser.add_argument('-which_set',
                        type=str,
                        default='test',
                        help='Step')
    parser.add_argument('--num_iter',
                        '-nit',
                        type=int,
                        default=10,
                        help='Max number of iterations.')
    parser.add_argument('-test_from_0_255',
                        type=bool,
                        default=False,
                        help='Whether to train from images within 0-255 range')

    args = parser.parse_args()

    inference(args.dataset, args.segmentation_net, which_set=args.which_set,
              num_iter=int(args.num_iter), savepath=SAVEPATH, loadpath=LOADPATH,
              test_from_0_255=args.test_from_0_255)

if __name__ == "__main__":
    main()
