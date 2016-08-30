import argparse

import numpy as np
import theano
import theano.tensor as T
from theano import config
import lasagne

from models.fcn8_void import buildFCN8
from dataset_loaders.images.camvid import CamvidDataset
from data_loader import load_data

from metrics2 import jaccard, accuracy

_FLOATX = config.floatX


def test(convmodel_name, dataset, padding=0,
         savepath="/Tmp/romerosa/itinf/models/"):

    # Define symbolic variables
    input_var = T.tensor4('input_var')
    target_var = T.ivector('target_var')

    # Build dataset iterator
    _, _, test_iter = load_data(dataset, one_hot=False,
                                train_crop_size=None)

    n_batches_test = test_iter.get_n_batches()
    n_classes = test_iter.get_n_classes()
    void_labels = test_iter.get_void_labels()

    # Build convolutional model
    if convmodel_name == "fcn8":
        convmodel = buildFCN8(3, input_var=input_var, trainable=True,
                              n_classes=n_classes, load_weights=False)
    else:
        raise ValueError("Unknown teacher network")

    # Load parameters
    savepath = savepath + dataset + "/"
    with np.load(savepath + 'fcn8_model.npz',) as f:
        param_values = [f['arr_%d' % i]
                        for i in range(len(f.files))]
        nlayers = len(lasagne.layers.get_all_params(convmodel))
        lasagne.layers.set_all_param_values(convmodel,
                                            param_values[:nlayers])

    print "Defining and compiling test functions"
    test_prediction = lasagne.layers.get_output(convmodel, deterministic=True)

    test_prediction_dimshuffle = test_prediction.dimshuffle((0, 2, 3, 1))
    sh = test_prediction_dimshuffle.shape
    test_prediction_2D = \
        test_prediction_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    test_acc = accuracy(test_prediction_2D, target_var, void_labels)
    test_jacc = jaccard(test_prediction_2D, target_var, n_classes)

    val_fn = theano.function([input_var, target_var], [test_acc,
                                                       test_jacc])

    # Iterate over test and compute metrics
    print "Testing"
    acc_test_tot = 0
    jacc_num_test_tot = np.zeros((1, n_classes))
    jacc_denom_test_tot = np.zeros((1, n_classes))
    for i in range(n_batches_test):
        # Get minibatch
        X_test_batch, L_test_batch = test_iter.next()
        L_test_batch = np.reshape(L_test_batch,
                                  np.prod(L_test_batch.shape))

        # Test step
        acc_test, jacc_test = val_fn(X_test_batch, L_test_batch)
        jacc_num_test, jacc_denom_test = jacc_test

        acc_test_tot += acc_test
        jacc_num_test_tot += jacc_num_test
        jacc_denom_test_tot += jacc_denom_test

    acc_test = acc_test_tot/n_batches_test
    jacc_test = np.mean(jacc_num_test_tot / jacc_denom_test_tot)

    out_str = "FINAL MODEL: acc test %f, jacc test %f"
    out_str = out_str % (acc_test,
                         jacc_test)
    print out_str


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-model',
                        default='fcn8',
                        help='Name of pre-trained model.')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-padding',
                        type=int,
                        default=92,
                        help='Padding to be added to he images')

    args = parser.parse_args()

    test(args.model, args.dataset, int(args.padding))

if __name__ == "__main__":
    main()
