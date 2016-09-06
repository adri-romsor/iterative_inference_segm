import argparse
import os
from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import theano
import theano.tensor as T
from theano import config

import lasagne
from lasagne.objectives import squared_error

from data_loader import load_data
from metrics import accuracy, jaccard
from models.DAE_h import buildDAE
from models.fcn8 import buildFCN8
from helpers import save_img

_FLOATX = config.floatX
_EPSILON = 10e-8

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


def inference(dataset, learn_step=0.005, num_iter=500,
              training_loss='squared_error', layer_h=['pool5'],
              n_filters=64, noise=0.1, conv_before_pool=1, additional_pool=0,
              dropout=0., skip=False, unpool_type='standard', from_gt=True,
              save_perstep=False, which_set='test',
              savepath=None, loadpath=None):
    #
    # Define symbolic variables
    #
    input_x_var = T.tensor4('input_x_var')
    input_h_var = []
    name = ''
    for l in layer_h:
        input_h_var += [T.tensor4()]
        name += ('_'+l)
    y_hat_var = T.tensor4('pred_y_var')
    target_var = T.ivector('target_var')
    target_var_4D = T.itensor4('target_var_4D')

    #
    # Build dataset iterator
    #
    if which_set == 'train':
        test_iter, _, _ = load_data(dataset, train_crop_size=None,
                                    one_hot=True, batch_size=[10, 10, 10])
    elif which_set == 'valid':
        _, test_iter, _ = load_data(dataset, train_crop_size=None,
                                    one_hot=True, batch_size=[10, 10, 10])
    if which_set == 'test':
        _, _, test_iter = load_data(dataset, train_crop_size=None,
                                    one_hot=True, batch_size=[10, 10, 10])

    n_batches_test = test_iter.get_n_batches()
    n_classes = test_iter.get_n_classes()
    void_labels = test_iter.get_void_labels()

    #
    # Prepare load/save directories
    #
    exp_name = '_'.join(layer_h)
    exp_name += '_f' + str(n_filters) + 'c' + str(conv_before_pool) + \
        'p' + str(additional_pool) + '_z' + str(noise)
    exp_name += '_' + training_loss + ('_skip' if skip else '')
    exp_name += ('_fromgt' if from_gt else '_fromfcn8')
    exp_name += '_' + unpool_type + ('_dropout' + str(dropout) if
                                     dropout > 0. else '')

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
    # Build FCN8  with pre-trained weights up to layer_h + prediction
    fcn = buildFCN8(3, input_var=input_x_var,
                    n_classes=n_classes,
                    void_labels=void_labels,
                    trainable=False, load_weights=True,
                    layer=layer_h+['probs_dimshuffle'])

    # Build DAE with pre-trained weights
    dae = buildDAE(input_h_var, y_hat_var, n_classes, layer_h,
                   noise, n_filters, conv_before_pool, additional_pool,
                   dropout=dropout, trainable=True, void_labels=void_labels,
                   skip=skip, unpool_type=unpool_type, load_weights=True,
                   path_weights=loadpath, model_name='dae_model.npz')

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
    y_hat_dimshuffle = y_hat_var.dimshuffle((0, 2, 3, 1))
    sh = y_hat_dimshuffle.shape
    y_hat_2D = y_hat_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # derivative of energy wrt input
    de = - (pred_dae - y_hat_var)

    # function to compute de
    de_fn = theano.function(input_h_var+[y_hat_var], de)

    loss = squared_error(pred_dae, target_var_4D).mean(axis=1)
    mask = target_var_4D.sum(axis=1)
    loss = loss * mask
    loss = loss.sum()/mask.sum()
    loss_fn = theano.function(input_h_var+[y_hat_var, target_var_4D], loss)
    pred_dae_fn = theano.function(input_h_var+[y_hat_var], pred_dae)

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
    acc_tot_dae = 0
    jacc_tot_dae = 0
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
        pred_test_batch = pred_fcn_fn(X_test_batch)
        Y_test_batch = pred_test_batch[-1]
        H_test_batch = pred_test_batch[:-1]

        # Compute metrics before iterative inference
        acc_old, jacc_old = val_fn(Y_test_batch, L_test_target)
        acc_tot_old += acc_old
        jacc_tot_old += jacc_old
        Y_test_batch_old = Y_test_batch

        Y_test_batch_dae = pred_dae_fn(*(H_test_batch+[Y_test_batch]))
        acc_dae, jacc_dae = val_fn(Y_test_batch_dae, L_test_target)
        acc_tot_dae += acc_dae
        jacc_tot_dae += jacc_dae

        # Iterative inference
        for it in range(num_iter):
            rec_loss = loss_fn(*(H_test_batch+[Y_test_batch,
                                               L_test_batch[:, :-1, :, :]]))

            print rec_loss
            grad = de_fn(*(H_test_batch+[Y_test_batch]))

            # hist = np.histogram(grad, 10)
            # print hist

            Y_test_batch = Y_test_batch - learn_step * grad

            if save_perstep:
                # Save images
                save_img(X_test_batch, L_test_batch.argmax(1), Y_test_batch,
                         Y_test_batch_old, savepath, n_classes,
                         'batch' + str(i) + '_' + 'step' + str(it),
                         void_labels)

            norm = np.linalg.norm(grad, axis=1).mean()
            if norm < _EPSILON:
                break
            # print norm
            acc_iter, jacc_iter = val_fn(Y_test_batch, L_test_target)
            print acc_iter, np.mean(jacc_iter[0, :] / jacc_iter[1, :])

        # Compute metrics
        acc, jacc = val_fn(Y_test_batch, L_test_target)

        acc_tot += acc
        jacc_tot += jacc

        jacc_perclass_old = jacc_tot_old[0, :]/jacc_tot_old[1, :]
        jacc_perclass = jacc_tot[0, :]/jacc_tot[1, :]
        jacc_perclass_dae = jacc_tot_dae[0, :]/jacc_tot_dae[1, :]

        info_str = "    fcn8 acc %f, iter acc %f, fcn8 jacc %f, iter jacc %f"
        info_str += ", dae acc  %f, dae jacc % f"
        info_str = info_str % (acc_tot_old,
                               acc_tot,
                               np.mean(jacc_tot_old[0, :]/jacc_tot_old[1, :]),
                               np.mean(jacc_tot[0, :] / jacc_tot[1, :]),
                               acc_tot_dae,
                               np.mean(jacc_tot_dae[0, :]/jacc_tot_dae[1, :])
                               )
        print info_str

        print jacc_perclass_old
        print jacc_perclass
        print jacc_perclass_dae
        print test_iter.get_mask_labels()

        if not save_perstep:
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
    parser = argparse.ArgumentParser(description='Iterative inference.')

    parser.add_argument('-dataset',
                        type=str,
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-step',
                        type=float,
                        default=.1,
                        help='step')
    parser.add_argument('--num_iter',
                        '-ne',
                        type=int,
                        default=10,
                        help='Max number of iterations')
    parser.add_argument('-training_loss',
                        type=str,
                        default='squared_error',
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
                        default='standard',
                        help='Unpooling type - standard or trackind')
    parser.add_argument('-from_gt',
                        type=bool,
                        default=True,
                        help='Whether to train from GT (true) or fcn' +
                        'output (False)')
    parser.add_argument('-save_perstep',
                        type=bool,
                        default=True,
                        help='Save new segmentations after each step update')
    parser.add_argument('-which_set',
                        type=str,
                        default='test',
                        help='Inference set')
    args = parser.parse_args()

    inference(args.dataset, float(args.step), int(args.num_iter),
              args.training_loss, args.layer_h, noise=args.noise,
              n_filters=args.n_filters, conv_before_pool=args.conv_before_pool,
              additional_pool=args.additional_pool, dropout=args.dropout,
              skip=args.skip, unpool_type=args.unpool_type,
              from_gt=args.from_gt, save_perstep=args.save_perstep,
              which_set=args.which_set, savepath=SAVEPATH, loadpath=LOADPATH)


if __name__ == "__main__":
    main()
