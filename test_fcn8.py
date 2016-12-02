#!/usr/bin/env python
import argparse
from getpass import getuser

import numpy as np

import theano
import theano.tensor as T
from theano import config
import lasagne

from models.fcn8 import buildFCN8
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
else:
    raise ValueError('Unknown user : {}'.format(getuser()))


def test(dataset, which_set='test', data_aug=False,
         savepath=None, loadpath=None):

    #
    # Define symbolic variables
    #
    input_x_var = T.tensor4('input_var')
    target_var_3D = T.itensor3('target_var_4D')

    #
    # Build dataset iterator
    #
    if which_set == 'train':
        test_iter, _, _ = load_data(dataset, train_crop_size=None,
                                    one_hot=False, batch_size=[10, 10, 10])
    elif which_set == 'valid':
        _, test_iter, _ = load_data(dataset, train_crop_size=None,
                                    one_hot=False, batch_size=[10, 10, 10])
    if which_set == 'test':
        _, _, test_iter = load_data(dataset, train_crop_size=None,
                                    one_hot=False, batch_size=[10, 10, 10])

    colors = test_iter.get_cmap_values()
    n_batches_test = test_iter.get_n_batches()
    n_classes = test_iter.get_n_classes()
    void_labels = test_iter.get_void_labels()
    nb_in_channels = test_iter.data_shape[0]
    void = n_classes if any(void_labels) else n_classes+1

    #
    # Prepare load/save directories
    #
    exp_name = 'fcn8'

    #
    # Build networks
    #
    print 'Building networks'
    # Build FCN8  with pre-trained weights up to layer_h + prediction
    fcn = buildFCN8(nb_in_channels, input_var=input_x_var,
                    n_classes=n_classes,
                    void_labels=void_labels,
                    trainable=False, load_weights=True,
                    layer=['probs_dimshuffle'],
                    pascal=(dataset == 'pascal'),
                    path_weights=WEIGHTS_PATH+dataset+'/fcn8_model.npz')

    #
    # Define and compile theano functions
    #
    print "Defining and compiling test functions"
    test_prediction = lasagne.layers.get_output(fcn, deterministic=True)[0]

    test_prediction_dimshuffle = test_prediction.dimshuffle((0, 2, 3, 1))
    sh = test_prediction_dimshuffle.shape
    test_prediction_2D = \
        test_prediction_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # Reshape iterative inference output to b01,c
    target_var_2D = target_var_3D.flatten()

    test_acc = accuracy(test_prediction_2D, target_var_2D, void_labels)
    test_jacc = jaccard(test_prediction_2D, target_var_2D, n_classes)

    val_fn = theano.function([input_x_var, target_var_3D], [test_acc,
                                                            test_jacc])
    pred_fcn_fn = theano.function([input_x_var], test_prediction)

    # Iterate over test and compute metrics
    print "Testing"
    acc_test_tot = 0
    jacc_num_test_tot = np.zeros((1, n_classes))
    jacc_denom_test_tot = np.zeros((1, n_classes))
    for i in range(n_batches_test):
        # Get minibatch
        X_test_batch, L_test_batch = test_iter.next()
        Y_test_batch = pred_fcn_fn(X_test_batch)

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

    out_str = "FINAL MODEL: acc test %f, jacc test .1%f"
    out_str = out_str % (acc_test,
                         jacc_test)
    print out_str

    print ">>> Per class jaccard:"
    labs = test_iter.get_mask_labels()

    for i in range(len(labs)-len(void_labels)):
        class_str = '    ' + labs[i] + ' : %f'
        class_str = class_str % (jacc_per_class[i])
        print class_str


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')

    args = parser.parse_args()

    test(args.dataset,savepath=SAVEPATH, loadpath=LOADPATH)

if __name__ == "__main__":
    main()
