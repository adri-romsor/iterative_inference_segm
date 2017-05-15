#!/usr/bin/env python2

import os
import argparse
import time
from getpass import getuser
from distutils.dir_util import copy_tree
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano import config
import lasagne
from lasagne.regularization import regularize_network_params
from lasagne.nonlinearities import softmax
from lasagne.layers import Pool2DLayer, Deconv2DLayer, InputLayer, ElemwiseSumLayer, DilatedConv2DLayer
from layers.mylayers import CroppingLayer
from lasagne.objectives import squared_error as squared_error_L

from data_loader import load_data
from metrics import crossentropy, entropy, squared_error_h, squared_error, jaccard
from models.fcn8 import buildFCN8
# from models.FCDenseNet import build_fcdensenet
from models.DAE_h import buildDAE
from models.fcn8_dae import buildFCN8_DAE
from models.contextmod_dae import buildDAE_contextmod
from layers.mylayers import DePool2D
from helpers import build_experiment_name
# imports for keras
from keras.layers import Input
from keras.models import Model
from ResUNet.model import assemble_model
from ResUNet.preprocessing import build_preprocessing
from ResUNet.blocks import (bottleneck,
                            basic_block,
                            basic_block_mp)

_FLOATX = config.floatX
if getuser() == 'romerosa':
    SAVEPATH = '/Tmp/romerosa/itinf/models/'
    LOADPATH = '/data/lisatmp4/romerosa/itinf/models/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/itinf/models/'
elif getuser() == 'jegousim':
    SAVEPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    LOADPATH = '/data/lisatmp4/jegousim/iterative_inference/'
    WEIGHTS_PATH = '/data/lisatmp4/romerosa/rnncnn/fcn8_model.npz'
elif getuser() == 'drozdzam':
    SAVEPATH = '/Tmp/drozdzam/itinf/models/'
    LOADPATH = '/data/lisatmp4/erraqabi/iterative_inference/models/'
    WEIGHTS_PATH = LOADPATH
elif getuser() == 'erraqaba':
    SAVEPATH = '/Tmp/erraqaba/iterative_inference/models/'
    LOADPATH = '/data/lisatmp4/erraqabi/iterative_inference/models/'
    WEIGHTS_PATH = LOADPATH
else:
    raise ValueError('Unknown user : {}'.format(getuser()))


def train(dataset, segm_net, learning_rate=0.005, lr_anneal=1.0,
          weight_decay=1e-4, num_epochs=500, max_patience=100,
          optimizer='rmsprop', training_loss=['squared_error'],
          batch_size=[10, 1, 1], ae_h=False,
          dae_dict_updates={}, data_augmentation={},
          savepath=None, loadpath=None, resume=False, train_from_0_255=False,
          lmb=1):

    #
    # Update DAE parameters
    #
    dae_dict = {'kind': 'fcn8',
                'dropout': 0.0,
                'skip': True,
                'unpool_type': 'standard',
                'n_filters': 64,
                'conv_before_pool': 1,
                'additional_pool': 0,
                'concat_h': ['input'],
                'noise': 0.0,
                'from_gt': True,
                'temperature': 1.0,
                'path_weights': '',
                'layer': 'probs_dimshuffle',
                'exp_name': ''}

    dae_dict.update(dae_dict_updates)

    #
    # Prepare load/save directories
    #
    exp_name = build_experiment_name(segm_net,
                                     training_loss=training_loss,
                                     data_aug=bool(data_augmentation),
                                     learning_rate=learning_rate,
                                     lr_anneal=lr_anneal,
                                     weight_decay=weight_decay,
                                     optimizer=optimizer, ae_h=ae_h,
                                     **dae_dict)
    if savepath is None:
        raise ValueError('A saving directory must be specified')

    savepath = os.path.join(savepath, dataset, exp_name)
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
    # Define symbolic variables
    #
    input_x_var = T.tensor4('input_x_var')  # tensor for input image batch
    input_mask_var = T.tensor4('input_mask_var')  # tensor for segmentation bach (input dae)
    input_concat_h_vars = [T.tensor4()] * len(dae_dict['concat_h'])  # tensor for hidden repr batch (input dae)
    target_var = T.tensor4('target_var')  # tensor for target batch
    lr = theano.shared(np.float32(learning_rate), 'learning_rate')

    #
    # Build dataset iterator
    #
    if dataset == 'em':
        data_augmentation = {'rotation_range':25,
                             'shear_range':0.41,
                             'horizontal_flip':True,
                             'vertical_flip':True,
                             'fill_mode':'reflect',
                             'spline_warp':True,
                             'warp_sigma':10,
                             'warp_grid_size':3}

    train_iter, val_iter, _ = load_data(dataset,
                                        data_augmentation,
                                        one_hot=True,
                                        batch_size=batch_size,
                                        return_0_255=train_from_0_255,
                                        )

    n_batches_train = train_iter.nbatches
    n_batches_val = val_iter.nbatches
    n_classes = train_iter.non_void_nclasses
    void_labels = train_iter.void_labels
    nb_in_channels = train_iter.data_shape[0]
    void = n_classes if any(void_labels) else n_classes+1

    #
    # Build networks
    #

    # Check that model and dataset get along
    print 'Checking options'
    assert (segm_net == 'fcn8' and dataset == 'camvid') or \
        (segm_net == 'densenet' and dataset == 'camvid') or \
        (segm_net == 'fcn_fcresnet' and dataset == 'em') or \
        (segm_net == 'resunet' and dataset == 'em')

    # Build segmentation network
    print 'Building segmentation network'
    if segm_net == 'fcn8':
        fcn = buildFCN8(nb_in_channels, input_x_var, n_classes=n_classes,
                        void_labels=void_labels,
                        path_weights=WEIGHTS_PATH+dataset+'/fcn8_model.npz',
                        load_weights=True, layer=dae_dict['concat_h']+[dae_dict['layer']])
        padding = 100
    elif segm_net == 'densenet':
        fcn = build_fcdensenet(input_x_var, nb_in_channels=nb_in_channels,
                                n_classes=n_classes, layer=dae_dict['concat_h'])
        padding = 0
    elif segm_net == 'resunet':
        print("-")*10
        print("ResUNet is the best!")
        print("-")*10
        preprocessing_kwargs = OrderedDict((
            ('img_shape', (1, None, None)),
            ('regularize_weights', None),
            ('nb_filter', 16),
            ('kernel_size', 3),
            ('nb_layers', 4),
            ('pre_unet', True),
            ('output_nb_filter', 1)
            ))
        resunet_model_kwargs = OrderedDict((
            ('input_shape', (1, None, None)),
            ('num_classes', 2),
            ('input_num_filters', 32),
            ('main_block_depth', [3, 8, 10, 3]),
            ('num_main_blocks', 3),
            ('num_init_blocks', 2),
            ('weight_decay', None),
            ('dropout', 0.5),
            ('short_skip', True),
            ('long_skip', True),
            ('long_skip_merge_mode', 'sum'),
            ('use_skip_blocks', False),
            ('relative_num_across_filters', 1),
            ('mainblock', bottleneck),
            ('initblock', basic_block_mp)
            ))
        # build preprocessort
        prep_model = build_preprocessing(**preprocessing_kwargs)
        # build resnet
        resunet = assemble_model(**resunet_model_kwargs)
        # setup model (preprocessor + resunet)
        inputs = Input(shape=preprocessing_kwargs['img_shape'])
        out_prep = prep_model(inputs)
        out_model = resunet(out_prep)
        model = Model(input=inputs, output=out_model)
        # load weights
        model.load_weights("/data/lisatmp4/romerosa/itinf/models/em/best_weights.hdf5")
        print("-")*10
        print ("We are done!")
        print("-")*10
    elif segm_net == 'fcn_fcresnet':
        raise NotImplementedError
    else:
        raise ValueError

    # Build DAE network
    print 'Building DAE network'

    if ae_h and dae_dict['kind'] != 'standard':
        raise ValueError('Plug&Play not implemented for ' + dae_dict['kind'])
    if ae_h and 'pool' not in dae_dict['concat_h'][-1]:
        raise ValueError('Plug&Play version needs concat_h to be different than input')
    ae_h = ae_h and 'pool' in dae_dict['concat_h'][-1]

    if dae_dict['kind'] == 'standard':
        dae = buildDAE(input_concat_h_vars, input_mask_var, n_classes,
                       nb_features_to_concat=fcn[0].output_shape[1], padding=padding,
                       trainable=True,
                       void_labels=void_labels, load_weights=resume,
                       path_weights=loadpath, model_name='dae_model_best.npz',
                       out_nonlin=softmax, concat_h=dae_dict['concat_h'],
                       noise=dae_dict['noise'], n_filters=dae_dict['n_filters'],
                       conv_before_pool=dae_dict['conv_before_pool'],
                       additional_pool=dae_dict['additional_pool'],
                       dropout=dae_dict['dropout'], skip=dae_dict['skip'],
                       unpool_type=dae_dict['unpool_type'], ae_h=ae_h)
    elif dae_dict['kind'] == 'fcn8':
        dae = buildFCN8_DAE(input_concat_h_vars, input_mask_var, n_classes,
                            nb_in_channels=n_classes, trainable=True,
                            load_weights=resume, pretrained=True, pascal=True,
                            concat_h=dae_dict['concat_h'], noise=dae_dict['noise'],
                            dropout=dae_dict['dropout'],
                            path_weights=os.path.join('/'.join(loadpath.split('/')[:-1]),
                            dae_dict['path_weights']),
                            model_name='dae_model_best.npz')
    elif dae_dict['kind'] == 'contextmod':
        dae = buildDAE_contextmod(input_concat_h_vars, input_mask_var, n_classes,
                                  path_weights=loadpath,
                                  model_name='dae_model.npz',
                                  trainable=True, load_weights=resume,
                                  out_nonlin=softmax, noise=dae_dict['noise'],
                                  concat_h=dae_dict['concat_h'])
    else:
        raise ValueError('Unknown dae kind')

    #
    # Define and compile theano functions
    #

    # training functions
    print "Defining and compiling training functions"

    # fcn prediction
    fcn_prediction = lasagne.layers.get_output(fcn, deterministic=True, batch_norm_use_averages=False)

    # select prediction layers (pooling and upsampling layers)
    dae_all_lays = lasagne.layers.get_all_layers(dae)
    if dae_dict['kind'] != 'contextmod':
        dae_lays = [l for l in dae_all_lays
                    if isinstance(l, Pool2DLayer) or
                    isinstance(l, CroppingLayer) or
                    isinstance(l, ElemwiseSumLayer) or
                    l == dae_all_lays[-1]]
        # dae_lays = dae_lays[::2]
    else:
        dae_lays = [l for l in dae_all_lays if isinstance(l, DilatedConv2DLayer)]

    if ae_h:
        h_ae_idx = [i for i, el in enumerate(dae_lays) if el.name == 'h_to_recon'][0]
        h_hat_idx = [i for i, el in enumerate(dae_lays) if el.name == 'h_hat'][0]

    # predictions
    dae_prediction_all = lasagne.layers.get_output(dae_lays)
    dae_prediction = dae_prediction_all[-1]
    dae_prediction_h = dae_prediction_all[:-1]

    test_dae_prediction_all = lasagne.layers.get_output(dae_lays,
                                                        deterministic=True)
    test_dae_prediction = test_dae_prediction_all[-1]
    test_dae_prediction_h = test_dae_prediction_all[:-1]

    # fetch h and h_hat if needed
    if ae_h:
        h = dae_prediction_all[h_ae_idx]
        h_hat = dae_prediction_all[h_hat_idx]
        h_test = test_dae_prediction_all[h_ae_idx]
        h_hat_test = test_dae_prediction_all[h_hat_idx]

    # loss
    loss = 0
    test_loss = 0

    # Convert DAE prediction to 2D
    dae_prediction_2D = dae_prediction.dimshuffle((0, 2, 3, 1))
    sh = dae_prediction_2D.shape
    dae_prediction_2D = dae_prediction_2D.reshape((T.prod(sh[:3]), sh[3]))

    test_dae_prediction_2D = test_dae_prediction.dimshuffle((0, 2, 3, 1))
    sh = test_dae_prediction_2D.shape
    test_dae_prediction_2D = test_dae_prediction_2D.reshape((T.prod(sh[:3]),
                                                            sh[3]))
    # Convert target to 2D
    target_var_2D = target_var.dimshuffle((0, 2, 3, 1))
    sh = target_var_2D.shape
    target_var_2D = target_var_2D.reshape((T.prod(sh[:3]), sh[3]))

    if 'crossentropy' in training_loss:
        # Compute loss
        loss += crossentropy(dae_prediction_2D, target_var_2D, void_labels,
                             one_hot=True)
        test_loss += crossentropy(test_dae_prediction_2D, target_var_2D,
                                  void_labels, one_hot=True)

    test_mse_loss = squared_error(test_dae_prediction, target_var, void)
    if 'squared_error' in training_loss:
        mse_loss = squared_error(dae_prediction, target_var, void)
        loss += lmb*mse_loss
        test_loss += lmb*test_mse_loss

    # Add intermediate losses
    if 'squared_error_h' in training_loss:
        # extract input layers and create dictionary
        dae_input_lays = [l for l in dae_all_lays if isinstance(l, InputLayer)]
        inputs = {dae_input_lays[0]: target_var[:, :void, :, :], dae_input_lays[-1]:target_var[:, :void, :, :]}
        for idx, val in enumerate(input_concat_h_vars):
            inputs[dae_input_lays[idx+1]] = val

        test_dae_prediction_all_gt = lasagne.layers.get_output(dae_lays,
                                                               inputs=inputs,
                                                               deterministic=True)
        test_dae_prediction_h_gt = test_dae_prediction_all_gt[:-1]

        loss += squared_error_h(dae_prediction_h, test_dae_prediction_h_gt)
        test_loss += squared_error_h(test_dae_prediction_h, test_dae_prediction_h_gt)

    # compute jaccard
    jacc = jaccard(dae_prediction_2D, target_var_2D, n_classes, one_hot=True)
    test_jacc = jaccard(test_dae_prediction_2D, target_var_2D, n_classes, one_hot=True)

    # if reconstructing h add the corresponding loss terms
    if ae_h:
        loss += squared_error_L(h, h_hat).mean()
        test_loss += squared_error_L(h_test, h_hat_test).mean()


    # network parameters
    params = lasagne.layers.get_all_params(dae, trainable=True)

    # optimizer
    if optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=lr)
    elif optimizer == 'adam':
        updates = lasagne.updates.adam(loss, params, learning_rate=lr)
    else:
        raise ValueError('Unknown optimizer')

    # functions
    train_fn = theano.function(input_concat_h_vars + [input_mask_var, target_var],
                               loss, updates=updates)
    fcn_fn = theano.function([input_x_var], fcn_prediction)
    val_fn = theano.function(input_concat_h_vars + [input_mask_var, target_var], [test_loss, test_jacc, test_mse_loss])

    err_train = []
    err_valid = []
    jacc_val_arr = []
    mse_val_arr = []
    patience = 0

    #
    # Train
    #

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

            #####uncomment if you want to control the feasability of pooling####
            # max_n_possible_pool = np.floor(np.log2(np.array(X_train_batch.shape[2:]).min()))
            # # check if we don't ask for more poolings than possible
            # assert n_pool+additional_pool < max_n_possible_pool
            ####################################################################

            # h prediction
            if segm_net == 'resunet':
                H_pred_batch = resunet.predict(X_train_batch)
            else:
                H_pred_batch = fcn_fn(X_train_batch)
            if dae_dict['from_gt']:
                Y_pred_batch = L_train_batch[:, :void, :, :]
            else:
                Y_pred_batch = H_pred_batch[-1]
            H_pred_batch = H_pred_batch[:-1]

            # Training step
            cost_train = train_fn(*(H_pred_batch + [Y_pred_batch, L_train_batch]))
            cost_train_tot += cost_train

        err_train += [cost_train_tot / n_batches_train]

        # Validation
        cost_val_tot = 0
        jacc_val_tot = 0
        mse_val_tot = 0
        for i in range(n_batches_val):
            # Get minibatch
            X_val_batch, L_val_batch = val_iter.next()
            L_val_batch = L_val_batch.astype(_FLOATX)

            # h prediction
            H_pred_batch = fcn_fn(X_val_batch)
            if dae_dict['from_gt']:
                Y_pred_batch = L_val_batch[:, :void, :, :]
            else:
                Y_pred_batch = H_pred_batch[-1]
            H_pred_batch = H_pred_batch[:-1]

            # Validation step
            cost_val, jacc_val, mse_val = val_fn(*(H_pred_batch + [Y_pred_batch, L_val_batch]))
            cost_val_tot += cost_val
            jacc_val_tot += jacc_val
            mse_val_tot += mse_val

        err_valid += [cost_val_tot / n_batches_val]
        jacc_val_arr += [np.mean(jacc_val_tot[0, :] / jacc_val_tot[1, :])]
        mse_val_arr += [mse_val_tot /  n_batches_val]

        out_str = "EPOCH %i: Avg epoch training cost train %f, cost val %f," + \
                  " jacc val %f, mse val % f took %f s"
        out_str = out_str % (epoch, err_train[epoch],
                             err_valid[epoch],
                             jacc_val_arr[epoch],
                             mse_val_arr[epoch],
                             time.time() - start_time)
        print out_str

        with open(os.path.join(savepath, "output.log"), "a") as f:
            f.write(out_str + "\n")

        # update learning rate
        lr.set_value(float(lr.get_value() * lr_anneal))

        # Early stopping and saving stuff
        if epoch == 0:
            best_err_val = err_valid[epoch]
            best_jacc_val = jacc_val_arr[epoch]
        elif epoch > 0 and jacc_val_arr[epoch] > best_jacc_val: #  and err_valid[epoch] < best_err_val
            best_err_val = err_valid[epoch]
            best_jacc_val = jacc_val_arr[epoch]
            patience = 0
            np.savez(os.path.join(savepath, 'dae_model_best.npz'),
                     *lasagne.layers.get_all_param_values(dae))
            np.savez(os.path.join(savepath, 'dae_errors_best.npz'),
                     err_train, err_valid, jacc_val_arr)
        else:
            patience += 1
            np.savez(os.path.join(savepath, 'dae_model_last.npz'),
                     *lasagne.layers.get_all_param_values(dae))
            np.savez(os.path.join(savepath, 'dae_errors_last.npz'),
                     err_train, err_valid, jacc_val_arr)

        # Finish training if patience has expired or max nber of epochs
        # reached
        if patience == max_patience or epoch == num_epochs - 1:
            # Copy files to loadpath
            if savepath != loadpath:
                print('Copying model and other training files to {}'.format(
                    loadpath))
                copy_tree(savepath, loadpath)
            # End
            print(' Training Done !')
            return


def main():
    parser = argparse.ArgumentParser(description='DAE training')
    parser.add_argument('-dataset',
                        type=str,
                        default='em',
                        help='Dataset.')
    parser.add_argument('-segmentation_net',
                        type=str,
                        default='resunet',
                        help='Segmentation network.')
    parser.add_argument('-train_dict',
                        type=dict,
                        default={'learning_rate': 0.001, 'lr_anneal': 0.99,
                                 'weight_decay': 0.0001, 'num_epochs': 1000,
                                 'max_patience': 100, 'optimizer': 'rmsprop',
                                 'batch_size': [10, 10, 10],
                                 'training_loss': ['crossentropy', 'squared_error']},
                        help='Training configuration')
    parser.add_argument('-dae_dict',
                        type=dict,
                        default={'kind': 'standard', 'dropout': 0.5, 'skip': True,
                                 'unpool_type': 'trackind', 'noise': 0,
                                 'concat_h': ['pool4'], 'from_gt': False,
                                 'n_filters': 64, 'conv_before_pool': 1,
                                 'additional_pool': 2, 'temperature': 1.0,
                                 'path_weights': '',  'layer': 'probs_dimshuffle',
                                 'exp_name' : ''},
                        help='DAE kind and parameters')
    parser.add_argument('-data_augmentation',
                        type=dict,
                        default={'crop_size': (224, 224),
                                 'horizontal_flip': True,
                                 'fill_mode':'constant'},
                        help='Dictionary of data augmentation to be used')
    parser.add_argument('-ae_h',
                        type=bool,
                        default=False,
                        help='Whether to reconstruct intermediate h')
    parser.add_argument('-train_from_0_255',
                        type=bool,
                        default=False,
                        help='Whether to train from images within 0-255 range')
    args = parser.parse_args()

    train(dataset=args.dataset, segm_net=args.segmentation_net,
          dae_dict_updates=args.dae_dict, data_augmentation=args.data_augmentation,
          train_from_0_255=args.train_from_0_255, ae_h=args.ae_h, resume=False,
          savepath=SAVEPATH, loadpath=LOADPATH, **args.train_dict)

if __name__ == "__main__":
    main()
