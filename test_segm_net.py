#!/usr/bin/env python
import argparse
from getpass import getuser

import numpy as np

import theano
import theano.tensor as T
from theano import config
import lasagne

from models.fcn8 import buildFCN8
from models.FCDenseNet import build_fcdensenet
from data_loader import load_data

from metrics import jaccard, accuracy

_FLOATX = config.floatX
if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/itinf/models/'
elif getuser() == 'jegousim':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/itinf/models/'
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


def test(dataset, segm_net, which_set='test', data_aug=False,
         savepath=None, loadpath=None, test_from_0_255=False):

    #
    # Define symbolic variables
    #
    input_x_var = T.tensor4('input_var')
    target_var = T.ivector('target_var')

    #
    # Build dataset iterator
    #
    data_iter = load_data(dataset, {}, one_hot=False, batch_size=[10, 10, 10],
                          return_0_255=test_from_0_255, which_set=which_set)

    colors = data_iter.cmap
    n_batches_test = data_iter.nbatches
    n_classes = data_iter.non_void_nclasses
    void_labels = data_iter.void_labels
    nb_in_channels = data_iter.data_shape[0]
    void = n_classes if any(void_labels) else n_classes+1

    #
    # Build segmentation network
    #
    print ' Building segmentation network'
    if segm_net == 'fcn8':
        fcn = buildFCN8(nb_in_channels, input_var=input_x_var,
                        n_classes=n_classes, void_labels=void_labels,
                        path_weights=WEIGHTS_PATH+dataset+'/fcn8_model.npz',
                        trainable=False, load_weights=True,
                        layer=['probs'])
    elif segm_net == 'densenet':
        fcn  = build_fcdensenet(input_x_var, nb_in_channels=nb_in_channels,
                                n_classes=n_classes, layer=[], output_d='2d')
    elif segm_net == 'fcn_fcresnet':
        raise NotImplementedError
    else:
        raise ValueError

    #
    # Define and compile theano functions
    #
    print "Defining and compiling test functions"
    test_prediction = lasagne.layers.get_output(fcn, deterministic=True, batch_norm_use_averages=False)[0]

    test_acc = accuracy(test_prediction, target_var, void_labels)
    test_jacc = jaccard(test_prediction, target_var, n_classes)

    val_fn = theano.function([input_x_var, target_var], [test_acc, test_jacc])
    pred_fcn_fn = theano.function([input_x_var], test_prediction)

    # Iterate over test and compute metrics
    print "Testing"
    acc_test_tot = 0
    jacc_num_test_tot = np.zeros((1, n_classes))
    jacc_denom_test_tot = np.zeros((1, n_classes))
    for i in range(n_batches_test):
        # Get minibatch
        X_test_batch, L_test_batch = data_iter.next()
        Y_test_batch = pred_fcn_fn(X_test_batch)
        L_test_batch = np.reshape(L_test_batch, np.prod(L_test_batch.shape))

        # Test step
        acc_test, jacc_test = val_fn(X_test_batch, L_test_batch)
        jacc_num_test, jacc_denom_test = jacc_test

        acc_test_tot += acc_test
        jacc_num_test_tot += jacc_num_test
        jacc_denom_test_tot += jacc_denom_test

        # Save images
        # save_img(X_test_batch, L_test_batch, Y_test_batch,
        #          savepath, n_classes, 'batch' + str(i),
    #              void_labels, colors)

    acc_test = acc_test_tot/n_batches_test
    jacc_per_class = jacc_num_test_tot / jacc_denom_test_tot
    jacc_per_class = jacc_per_class[0]
    jacc_test = np.mean(jacc_per_class)

    out_str = "FINAL MODEL: acc test %f, jacc test %f"
    out_str = out_str % (acc_test,
                         jacc_test)
    print out_str

    print ">>> Per class jaccard:"
    labs = data_iter.mask_labels

    for i in range(len(labs)-len(void_labels)):
        class_str = '    ' + labs[i] + ' : %f'
        class_str = class_str % (jacc_per_class[i])
        print class_str


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-segmentation_net',
                        type=str,
                        default='densenet',
                        help='Segmentation network.')
    parser.add_argument('-test_from_0_255',
                        type=bool,
                        default=False,
                        help='Whether to train from images within 0-255 range')

    args = parser.parse_args()

    test(args.dataset, args.segmentation_net, savepath=SAVEPATH, loadpath=LOADPATH,
         test_from_0_255=args.test_from_0_255)

if __name__ == "__main__":
    main()
