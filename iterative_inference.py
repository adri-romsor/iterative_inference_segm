import argparse
import os
import numpy as np

import theano
import theano.tensor as T
from theano import config

import lasagne

from data_loader import load_data
from metrics import accuracy, jaccard
from models.DAE_h import buildDAE
from models.fcn8_void import buildFCN8
from helpers import save_img

_FLOATX = config.floatX


def inference(dataset, layer_name=None, learn_step=0.005, num_iter=500,
              num_filters=[256], skip=False, filter_size=[3], savepath=None):

    # Define symbolic variables
    input_x_var = T.tensor4('input_x_var')
    input_h_var = []
    name = ''
    for l in layer_name:
        input_h_var += [T.tensor4()]
        name += ('_'+l)
    y_hat_var = T.tensor4('pred_y_var')
    input_dae_mask_var = T.tensor4('input_dae_mask_var')
    target_var = T.ivector('target_var')

    # Build dataset iterator
    _, _, test_iter = load_data(dataset, train_crop_size=None, one_hot=True)

    n_batches_test = test_iter.get_n_batches()
    n_classes = test_iter.get_n_classes()
    void_labels = test_iter.get_void_label()

    # Prepare saving directory
    savepath = savepath + dataset + "/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

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
                   void_labels=void_labels, skip=skip,
                   model_name=dataset+'/dae_model'+name+'.npz')

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

            if grad.min() == 0 and grad.max() == 0:
                break

        # Compute metrics
        acc, jacc = val_fn(Y_test_batch, L_test_target)

        acc_tot += acc
        jacc_tot += jacc

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


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        type=str,
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-layer_name',
                        type=list,
                        default=['pool1', 'pool3'],
                        help='All h to introduce to the DAE.')
    parser.add_argument('-step',
                        type=float,
                        default=0.001,
                        help='Step')
    parser.add_argument('--num_iter',
                        '-nit',
                        type=int,
                        default=500,
                        help='Max number of iterations.')
    parser.add_argument('-num_filters',
                        type=list,
                        default=[512],
                        help='All h to introduce to the DAE.')
    parser.add_argument('-skip',
                        type=bool,
                        default=False,
                        help='Whether to skip connections in the DAE.')
    parser.add_argument('--savepath',
                        '-sp',
                        type=str,
                        default='/Tmp/romerosa/itinf/img_plots/',
                        help='Path to save images')

    args = parser.parse_args()

    inference(args.dataset, args.layer_name, float(args.step),
              int(args.num_iter), args.num_filters, args.skip,
              savepath=args.savepath)

if __name__ == "__main__":
    main()
