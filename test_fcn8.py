import argparse
from getpass import getuser

import numpy as np

import theano
import theano.tensor as T
from theano import config
import lasagne

from models.fcn8_void import buildFCN8
from data_loader import load_data

from metrics2 import jaccard, accuracy

_FLOATX = config.floatX
if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/Tmp/romerosa/itinf/models/camvid/fcn8_model.npz'
elif getuser() == 'jegousim':
    SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    LOADPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/rnncnn/fcn8_model.npz'
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
    target_var_4D = T.itensor4('target_var_4D')

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
    test_prediction = lasagne.layers.get_output(fcn,
                                                deterministic=True)[0]

    test_prediction_dimshuffle = test_prediction.dimshuffle((0, 2, 3, 1))
    sh = test_prediction_dimshuffle.shape
    test_prediction_2D = \
        test_prediction_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # Reshape iterative inference output to b01,c
    target_var_dimshuffle = target_var_4D.dimshuffle((0, 2, 3, 1))
    sh2 = target_var_dimshuffle.shape
    target_var_2D = target_var_dimshuffle.reshape((T.prod(sh2[:3]), sh2[3]))

    test_acc = accuracy(test_prediction_2D, target_var_2D, void_labels)
    test_jacc = jaccard(test_prediction_2D, target_var_2D, n_classes)

    val_fn = theano.function([input_x_var, target_var_4D], [test_acc,
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
        save_img(X_test_batch, L_test_batch, Y_test_batch,
                 savepath, n_classes, 'batch' + str(i),
                 void_labels, colors)

    acc_test = acc_test_tot/n_batches_test
    jacc_test = np.mean(jacc_num_test_tot / jacc_denom_test_tot)

    out_str = "FINAL MODEL: acc test %f, jacc test %f"
    out_str = out_str % (acc_test,
                         jacc_test)
    print out_str


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='em',
                        help='Dataset.')
    parser.add_argument('-learning_rate',
                        default=0.001,
                        help='Learning Rate')
    parser.add_argument('-penal_cst',
                        default=0.0,
                        help='regularization constant')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=1000,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')
    parser.add_argument('-max_patience',
                        type=int,
                        default=100,
                        help='Max patience')
    parser.add_argument('-data_aug',
                        type=bool,
                        default=True,
                        help='use data augmentation')
    args = parser.parse_args()

    test(args.dataset, float(args.learning_rate),
         float(args.penal_cst), int(args.num_epochs), int(args.max_patience),
         data_aug=args.data_aug, savepath=SAVEPATH, loadpath=LOADPATH)

if __name__ == "__main__":
    main()
