import os
import argparse
import time

import numpy as np
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.objectives import squared_error
from lasagne.regularization import regularize_network_params

from data_loader import load_data
from metrics import crossentropy, entropy
from models.DAE_h import buildDAE
from models.fcn8_void import buildFCN8

_FLOATX = config.floatX


def train(dataset, learn_step=0.005,
          weight_decay=1e-4, num_epochs=500, max_patience=100,
          epsilon=.0, optimizer='rmsprop', training_loss='squared_error',
          layer_h='pool5', num_filters=[512], filter_size=[3],
          savepath='/Tmp/romerosa/itinf/models/', resume=False):

    # Define symbolic variables
    input_x_var = T.tensor4('input_x_var')
    input_mask_var = T.tensor4('input_mask_var')

    input_repr_var = []
    name = ''
    for l in layer_h:
        input_repr_var += [T.tensor4()]
        name += ('_'+l)

    # Build dataset iterator
    train_iter, val_iter, _ = load_data(dataset, train_crop_size=None,
                                        one_hot=True)

    n_batches_train = train_iter.get_n_batches()
    n_batches_val = val_iter.get_n_batches()
    n_classes = train_iter.get_n_classes()
    void_labels = train_iter.get_void_label()

    # Prepare saving directory
    savepath = savepath + dataset + "/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    name = '_' + layer_h

    # Build FCN
    print ' Building FCN network'
    fcn = buildFCN8(3, input_x_var, n_classes=n_classes,
                    void_labels=void_labels,
                    trainable=True, load_weights=True, layer=layer_h)

    # Build DAE network
    print ' Building DAE network'
    dae = buildDAE(input_repr_var, input_mask_var, n_classes,
                   layer_h, num_filters, filter_size, trainable=True,
                   load_weights=resume, void_labels=void_labels,
                   model_name=dataset+'/dae_model'+name+'.npz')

    # Define required theano functions for training and compile them
    print "Defining and compiling training functions"

    # prediction and loss
    fcn_prediction = lasagne.layers.get_output(fcn, deterministic=True)

    prediction = lasagne.layers.get_output(dae)
    if training_loss == 'crossentropy':
        loss = crossentropy(prediction, input_mask_var, void_labels)
    elif training_loss == 'squared_error':
        loss = squared_error(prediction, input_mask_var).mean()
    else:
        raise ValueError('Unknown training loss')

    loss += epsilon*entropy(prediction)

    # regularizers
    weightsl2 = regularize_network_params(
        dae, lasagne.regularization.l2)
    loss += weight_decay * weightsl2

    params = lasagne.layers.get_all_params(dae, trainable=True)

    # optimizer
    if optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params,
                                          learning_rate=learn_step)
    elif optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params,
                                       learning_rate=learn_step)
    else:
        raise ValueError('Unknown optimizer')

    # functions
    train_fn = theano.function(input_repr_var+[input_mask_var],
                               loss, updates=updates)
    fcn_fn = theano.function([input_x_var], fcn_prediction)

    # Define required theano functions for testing and compile them
    print "Defining and compiling test functions"
    # prediction and loss
    test_prediction = lasagne.layers.get_output(dae, deterministic=True)
    if training_loss == 'crossentropy':
        test_loss = crossentropy(test_prediction, input_mask_var, void_labels)
    elif training_loss == 'squared_error':
        test_loss = squared_error(test_prediction, input_mask_var).mean()
    else:
        raise ValueError('Unknown training loss')

    # functions
    val_fn = theano.function(input_repr_var+[input_mask_var], test_loss)

    err_train = []
    err_valid = []
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
            L_train_batch = L_train_batch.astype(_FLOATX)

            # h prediction
            X_pred_batch = fcn_fn(X_train_batch)

            # Training step
            cost_train = train_fn(X_pred_batch, L_train_batch)
            cost_train_tot += cost_train

        err_train += [cost_train_tot/n_batches_train]

        # Validation
        cost_val_tot = 0
        for i in range(n_batches_val):
            # Get minibatch
            X_val_batch, L_val_batch = val_iter.next()
            L_val_batch = L_val_batch.astype(_FLOATX)

            # h prediction
            X_pred_batch = fcn_fn(X_val_batch)

            # Validation step
            cost_val = val_fn(X_pred_batch, L_val_batch)
            cost_val_tot += cost_val

        err_valid += [cost_val_tot/n_batches_val]

        out_str = "EPOCH %i: Avg epoch training cost train %f, cost val %f" +\
            " took %f s"
        out_str = out_str % (epoch, err_train[epoch],
                             err_valid[epoch],
                             time.time()-start_time)
        print out_str

        with open(savepath + "output" + name + ".log", "a") as f:
            f.write(out_str + "\n")

        # Early stopping and saving stuff
        if epoch == 0:
            best_err_val = err_valid[epoch]
        elif epoch > 1 and err_valid[epoch] < best_err_val:
            best_err_val = err_valid[epoch]
            patience = 0
            np.savez(savepath + 'dae_model' + name + '.npz',
                     *lasagne.layers.get_all_param_values(dae))
            np.savez(savepath + 'dae_errors' + name + '.npz',
                     err_valid, err_train)
        else:
            patience += 1

        # Finish training if patience has expired or max nber of epochs
        # reached
        if patience == max_patience or epoch == num_epochs-1:
            # End
            return


def main():
    parser = argparse.ArgumentParser(description='Unet model training')
    parser.add_argument('-dataset',
                        default='camvid',
                        help='Dataset.')
    parser.add_argument('-learning_rate',
                        default=0.00001,
                        help='Learning rate')
    parser.add_argument('-weight_decay',
                        default=.0,
                        help='Weight decay')
    parser.add_argument('--num_epochs',
                        '-ne',
                        type=int,
                        default=2000,
                        help='Optional. Int to indicate the max'
                        'number of epochs.')
    parser.add_argument('--max_patience',
                        '-mp',
                        type=int,
                        default=100,
                        help='Optional. Int to indicate the max'
                        'patience.')
    parser.add_argument('-epsilon',
                        default=0.,
                        help='Entropy weight')
    parser.add_argument('-optimizer',
                        type=str,
                        default='adam',
                        help='Optional. Optimizer (adam or rmsprop)')
    parser.add_argument('-training_loss',
                        type=str,
                        default='squared_error',
                        help='Optional. Training loss')
    parser.add_argument('-layer_h',
                        type=list,
                        default=['pool1', 'pool3'],
                        help='layer_h')
    args = parser.parse_args()

    train(args.dataset, float(args.learning_rate),
          float(args.weight_decay), int(args.num_epochs),
          int(args.max_patience), float(args.epsilon),
          args.optimizer, args.training_loss, args.layer_h, resume=True)


if __name__ == "__main__":
    main()
