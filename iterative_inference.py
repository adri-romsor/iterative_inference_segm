import argparse
import numpy as np

import theano
import theano.tensor as T
from theano import config

import lasagne

from data_loader import load_data
from metrics import accuracy, jaccard
from models.DAE import buildDAE
from models.fcn8_void import buildFCN8

_FLOATX = config.floatX


def inference(dataset, layer_name=None, learn_step=0.005, num_iter=500):

    # Define symbolic variables
    input_fcn_var = T.tensor4('input_fcn_var')
    input_dae_mask_var = T.tensor4('input_dae_mask_var')
    infer_out_var = T.tensor4('infer_out_var')
    target_var = T.ivector('target_var')

    # Build dataset iterator
    train_iter, val_iter, test_iter = load_data(dataset)

    n_batches_test = test_iter.get_n_batches()
    n_classes = train_iter.get_n_classes()
    void_labels = train_iter.get_void_label()

    # Compute number of input channels of DAE
    if layer_name == 'input':
        n_input_dae = 3
        n_classes_dae = n_classes + (1 if void_labels else 0)
    else:
        raise ValueError('unknown input layer')

    print 'Building networks'
    # Build FCN8 with pre-trained weights
    fcn = buildFCN8(3, input_var=input_fcn_var,
                    n_classes=n_classes,
                    void_labels=void_labels,
                    trainable=False, load_weights=True)

    # Build DAE with pre-trained weights
    dae = buildDAE(input_fcn_var, input_dae_mask_var,
                   n_input_dae, n_classes_dae, filter_size=[64],
                   kernel_size=[3], trainable=False, load_weights=True)

    print "Defining and compiling theano functions"
    # Define required theano functions and compile them
    # predictions of fcn and dae
    pred_fcn = lasagne.layers.get_output(fcn, deterministic=True)
    pred_dae = lasagne.layers.get_output(dae, deterministic=True)

    # function to compute output of fcn
    pred_fcn_fn = theano.function([input_fcn_var], pred_fcn)

    # Reshape iterative inference output to b,01c
    infer_out_dimshuffle = infer_out_var.dimshuffle((0, 2, 3, 1))
    sh = infer_out_dimshuffle.shape
    infer_out_metrics = infer_out_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # derivative of energy wrt input
    de = - pred_dae - pred_fcn

    # function to compute de
    de_fn = theano.function([input_fcn_var,
                             input_dae_mask_var], de)

    # metrics to evaluate iterative inference
    test_acc = accuracy(infer_out_metrics, target_var, void_labels)
    test_jacc = jaccard(infer_out_metrics, target_var, n_classes)

    # functions to compute metrics
    val_fn = theano.function([infer_out_var, target_var],
                             [test_acc, test_jacc])

    print 'Start infering'
    acc_tot = 0
    jacc_tot = 0
    for i in range(n_batches_test):
        info_str = "Batch %d out of %d" % (i, n_batches_test)
        print info_str

        # Get minibatch
        X_test_batch, L_test_batch = test_iter.next()
        L_test_target = L_test_batch.argmax(1)
        L_test_target = np.reshape(L_test_target,
                                   np.prod(L_test_target.shape))
        L_test_target = L_test_target.astype('int32')

        # Compute fcn prediction
        pred = pred_fcn_fn(X_test_batch)

        # Iterative inference
        for it in range(num_iter):
            grad = de_fn(X_test_batch, L_test_batch.astype(_FLOATX))

            pred = pred - learn_step * grad

            if grad.min() == 0 and grad.max() == 0:
                break

        # Test step
        acc, jacc = val_fn(pred, L_test_target)

        acc_tot += acc
        jacc_tot += jacc

    acc_test = acc_tot/n_batches_test
    jacc_test = np.mean(jacc_tot[0, :] / jacc_tot[1, :])

    out_str = "FINAL MODEL: acc test % f, jacc test %f"
    out_str = out_str % (acc_test, jacc_test)
    print out_str


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-layer_name',
                        default='input',
                        help='Dataset.')
    parser.add_argument('-learning_rate',
                        default=0.0001,
                        help='Learning Rate')
    parser.add_argument('-penal_cst',
                        default=0.0,
                        help='regularization constant')
    parser.add_argument('--num_iter',
                        '-nit',
                        type=int,
                        default=200,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')

    args = parser.parse_args()

    inference(args.dataset, args.layer_name, float(args.learning_rate),
              int(args.num_iter))

if __name__ == "__main__":
    main()
