import os
import argparse
import time
from getpass import getuser

import numpy as np
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params

from data_loader import load_data
from models.fcn8 import buildFCN8
from metrics import jaccard, accuracy, crossentropy

_FLOATX = config.floatX

if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
else:
    SAVEPATH = '/data/lisatmp4/romerosa/rnncn/fcn8_model.npz'
print('Pretrained FCN8 loaded from ' + SAVEPATH)

def train(dataset, learn_step=0.005,
          weight_decay=1e-4, num_epochs=500,
          max_patience=100,
          savepath=SAVEPATH):

    # Define symbolic variables
    input_var = T.tensor4('input_var')
    target_var = T.ivector('target_var')

    # Build dataset iterator
    train_iter, val_iter, test_iter = load_data(dataset, one_hot=False)

    n_batches_train = train_iter.get_n_batches()
    n_batches_val = val_iter.get_n_batches()
    n_batches_test = test_iter.get_n_batches()
    n_classes = train_iter.get_n_classes()
    void_labels = train_iter.get_void_label()

    # Build convolutional model and load pre-trained parameters
    convmodel = buildFCN8(3, input_var=input_var, trainable=True,
                          n_classes=n_classes, load_weights=True, pascal=True)

    # Define required theano functions and compile them
    print "Defining and compiling training functions"
    prediction = lasagne.layers.get_output(convmodel)
    loss = crossentropy(prediction, target_var, void_labels)

    weightsl2 = regularize_network_params(
        convmodel, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

    params = lasagne.layers.get_all_params(convmodel, trainable=True)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=learn_step)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    print "Defining and compiling test functions"
    test_prediction = lasagne.layers.get_output(convmodel, deterministic=True)
    test_loss = crossentropy(test_prediction, target_var, void_labels)
    test_acc = accuracy(test_prediction, target_var, void_labels)
    test_jacc = jaccard(test_prediction, target_var, n_classes)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc,
                                                       test_jacc])

    # Prepare saving directory
    savepath = savepath + dataset + "/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    err_train = []
    err_valid = []
    acc_valid = []
    jacc_valid = []
    patience = 0

    # Training main loop
    print "Start training"
    for epoch in range(num_epochs):
        # Single epoch training and validation
        start_time = time.time()
        cost_train_tot = 0

        # Train
        for i in range(n_batches_train):
            # Get minibatch
            X_train_batch, L_train_batch = train_iter.next()
            L_train_batch = np.reshape(L_train_batch,
                                       np.prod(L_train_batch.shape))

            # Training step
            cost_train = train_fn(X_train_batch, L_train_batch)
            out_str = "cost %f" % (cost_train)
            cost_train_tot += cost_train

        err_train += [cost_train_tot/n_batches_train]

        # Validation
        cost_val_tot = 0
        acc_val_tot = 0
        jacc_val_tot = np.zeros((2, n_classes))
        for i in range(n_batches_val):
            # Get minibatch
            X_val_batch, L_val_batch = val_iter.next()
            L_val_batch = np.reshape(L_val_batch,
                                     np.prod(L_val_batch.shape))

            # Validation step
            cost_val, acc_val, jacc_val = val_fn(X_val_batch, L_val_batch)

            acc_val_tot += acc_val
            cost_val_tot += cost_val
            jacc_val_tot += jacc_val

        err_valid += [cost_val_tot/n_batches_val]
        acc_valid += [acc_val_tot/n_batches_val]
        jacc_valid += [np.mean(jacc_val_tot[0, :] /
                               jacc_val_tot[1, :])]

        out_str = "EPOCH %i: Avg epoch training cost train %f, cost val %f" +\
            ", acc val %f, jacc val %f took %f s"
        out_str = out_str % (epoch, err_train[epoch],
                             err_valid[epoch],
                             acc_valid[epoch],
                             jacc_valid[epoch],
                             time.time()-start_time)
        print out_str

        with open(savepath + "fcn8_new_output.log", "a") as f:
            f.write(out_str + "\n")

        # Early stopping and saving stuff
        if epoch == 0:
            best_jacc_val = jacc_valid[epoch]
        elif epoch > 1 and jacc_valid[epoch] > best_jacc_val:
            best_jacc_val = jacc_valid[epoch]
            patience = 0
            np.savez(savepath + 'fcn8_new_model.npz',
                     *lasagne.layers.get_all_param_values(convmodel))
            np.savez(savepath + "fcn8_new_errors.npz",
                     err_valid, err_train, acc_valid,
                     jacc_valid)
        else:
            patience += 1

        # Finish training if patience has expired or max nber of epochs
        # reached
        if patience == max_patience or epoch == num_epochs-1:
            # Load best model weights
            with np.load(savepath + 'fcn8_new_model.npz',) as f:
                param_values = [f['arr_%d' % i]
                                for i in range(len(f.files))]
            nlayers = len(lasagne.layers.get_all_params(convmodel))
            lasagne.layers.set_all_param_values(convmodel,
                                                param_values[:nlayers])
            # Test
            cost_test_tot = 0
            acc_test_tot = 0
            jacc_num_test_tot = np.zeros((1, n_classes))
            jacc_denom_test_tot = np.zeros((1, n_classes))
            for i in range(n_batches_test):
                # Get minibatch
                X_test_batch, L_test_batch = test_iter.next()
                L_test_batch = np.reshape(L_test_batch,
                                          np.prod(L_test_batch.shape))

                # Test step
                cost_test, acc_test, jacc_test = \
                    val_fn(X_test_batch, L_test_batch)
                jacc_num_test, jacc_denom_test = jacc_test

                acc_test_tot += acc_test
                cost_test_tot += cost_test
                jacc_num_test_tot += jacc_num_test
                jacc_denom_test_tot += jacc_denom_test

            err_test = cost_test_tot/n_batches_test
            acc_test = acc_test_tot/n_batches_test
            jacc_test = np.mean(jacc_num_test_tot / jacc_denom_test_tot)

            out_str = "FINAL MODEL: err test % f, acc test %f, jacc test %f"
            out_str = out_str % (err_test,
                                 acc_test,
                                 jacc_test)
            print out_str

            # End
            return


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-learning_rate',
                        default=0.0001,
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

    args = parser.parse_args()

    train(args.dataset, float(args.learning_rate),
          float(args.penal_cst), int(args.num_epochs), int(args.max_patience))

if __name__ == "__main__":
    main()
