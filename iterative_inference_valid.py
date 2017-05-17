#!/usr/bin/env python2

import argparse
import os
from getpass import getuser
from distutils.dir_util import copy_tree

import numpy as np
import theano
import theano.tensor as T
from theano import config

import lasagne
from lasagne.nonlinearities import softmax

from data_loader import load_data
from metrics import accuracy, jaccard, squared_error
from models.DAE_h import buildDAE
from models.fcn8 import buildFCN8
from models.fcn8_dae import buildFCN8_DAE
from models.FCDenseNet import build_fcdensenet
from models.contextmod_dae import buildDAE_contextmod
from helpers import save_img
from models.model_helpers import softmax4D
from helpers import build_experiment_name, print_results

# imports for keras
from keras.layers import Input
from keras.models import Model
from models.fcn_resunet_model import assemble_model
from models.fcn_resunet_preprocessing import build_preprocessing
from models.fcn_resunet_blocks import (bottleneck,
                                       basic_block,
                                       basic_block_mp)
from collections import OrderedDict

_FLOATX = config.floatX
_EPSILON = 10e-8

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
    LOADPATH = '/data/lisatmp4/drozdzam/itinf/models/'
    WEIGHTS_PATH = LOADPATH
elif getuser() == 'erraqaba':
    SAVEPATH = '/Tmp/erraqaba/iterative_inference/models/'
    LOADPATH = '/data/lisatmp4/erraqabi/iterative_inference/models/'
    WEIGHTS_PATH = LOADPATH
else:
    raise ValueError('Unknown user : {}'.format(getuser()))

_EPSILON = 1e-3


def inference(dataset, segm_net, learn_step=0.005, num_iter=500,
              dae_dict_updates= {}, training_dict={}, data_augmentation=False,
              which_set='test', ae_h=False, full_im_ft=False,
              savepath=None, loadpath=None, test_from_0_255=False):

    #
    # Update DAE parameters
    #
    dae_dict = {'kind': 'fcn8',
                'dropout': 0.0,
                'skip': True,
                'unpool_type':'standard',
                'n_filters': 64,
                'conv_before_pool': 1,
                'additional_pool': 0,
                'concat_h': ['input'],
                'noise': 0.0,
                'from_gt': True,
                'temperature': 1.0,
                'layer': 'probs_dimshuffle',
                'exp_name': '',
                'bn': 0}

    dae_dict.update(dae_dict_updates)

    #
    # Prepare load/save directories
    #
    exp_name = build_experiment_name(segm_net, data_aug=data_augmentation, ae_h=ae_h,
                                     **dict(dae_dict.items() + training_dict.items()))
    exp_name += '_ftsmall' if full_im_ft else ''

    if savepath is None:
        raise ValueError('A saving directory must be specified')

    savepath = os.path.join(savepath, dataset, exp_name, 'img_plots', str(learn_step),
                            which_set)
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
    input_concat_h_vars = [T.tensor4()] * len(dae_dict['concat_h'])  # tensor for hidden repr batch (input dae)
    y_hat_var = T.tensor4('pred_y_var')
    target_var = T.tensor4('target_var')  # tensor for target batch

    #
    # Build dataset iterator
    #
    data_iter = load_data(dataset, {}, one_hot=True, batch_size=[10, 5, 10],
                          return_0_255=test_from_0_255, which_set=which_set)

    colors = data_iter.cmap
    n_batches_test = data_iter.nbatches
    n_classes = data_iter.non_void_nclasses
    void_labels = data_iter.void_labels
    nb_in_channels = data_iter.data_shape[0]
    void = n_classes if any(void_labels) else n_classes+1

    #
    # Build networks
    #

    # Build segmentation network
    print 'Building segmentation network'
    if segm_net == 'fcn8':
        fcn = buildFCN8(nb_in_channels, input_var=input_x_var,
                        n_classes=n_classes, void_labels=void_labels,
                        path_weights=WEIGHTS_PATH+dataset+'/fcn8_model.npz',
                        trainable=False, load_weights=True,
                        layer=dae_dict['concat_h']+[dae_dict['layer']])
        padding = 100
    elif segm_net == 'densenet':
        fcn = build_fcdensenet(input_x_var, nb_in_channels=nb_in_channels,
                                n_classes=n_classes, layer=dae_dict['concat_h'])
        padding = 0
    elif segm_net == 'fcn_fcresnet':
        padding = 0
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
            ('initblock', basic_block_mp),
            # possible strings: input, initblock_d{0, 1}, mainblock_d{0, 1, 2, 3}
            ('hidden_outputs', ['mainblock_d1'])
            ))
        # build preprocessor
        prep_model = build_preprocessing(**preprocessing_kwargs)
        # build resnet
        resunet = assemble_model(**resunet_model_kwargs)
        # setup model (preprocessor + resunet)
        inputs = Input(shape=preprocessing_kwargs['img_shape'])
        out_prep = prep_model(inputs)
        out_model = resunet(out_prep)
        fcn = Model(input=inputs, output=out_model)
        # load weights
        fcn.load_weights("/data/lisatmp4/romerosa/itinf/models/em/best_weights.hdf5")
        print("-")*10
        print ("Resunet model loading done!")
        print("-")*10
    else:
        raise ValueError

    # Build DAE with pre-trained weights
    print 'Building DAE network'
    if dae_dict['kind'] == 'standard':
        if segm_net in ['fcn_fcresnet']:
            nb_features_to_concat=fcn.output_shape[0][1]
        else:
            nb_features_to_concat=fcn[0].output_shape[1]
        dae = buildDAE(input_concat_h_vars, y_hat_var, n_classes,
                       nb_features_to_concat=nb_features_to_concat,
                       padding=padding, trainable=True,
                       void_labels=void_labels, load_weights=True,
                       path_weights=loadpath, model_name='dae_model_best.npz',
                       out_nonlin=softmax, concat_h=dae_dict['concat_h'],
                       noise=dae_dict['noise'], n_filters=dae_dict['n_filters'],
                       conv_before_pool=dae_dict['conv_before_pool'],
                       additional_pool=dae_dict['additional_pool'],
                       dropout=dae_dict['dropout'], skip=dae_dict['skip'],
                       unpool_type=dae_dict['unpool_type'],
                       bn=dae_dict['bn'])
    elif dae_dict['kind'] == 'fcn8':
        dae = buildFCN8_DAE(input_concat_h_vars, y_hat_var, n_classes,
                            nb_in_channels=n_classes, path_weights=loadpath,
                            model_name='dae_model_best.npz', trainable=True,
                            load_weights=True, pretrained=True, pascal=False,
                            concat_h=dae_dict['concat_h'], noise=dae_dict['noise'])
    elif dae_dict['kind'] == 'contextmod':
        dae = buildDAE_contextmod(input_concat_h_vars, y_hat_var, n_classes,
                                  path_weights=loadpath,
                                  model_name='dae_model_best.npz',
                                  trainable=True, load_weights=True,
                                  out_nonlin=softmax, noise=dae_dict['noise'],
                                  concat_h=dae_dict['concat_h'])
    else:
        raise ValueError('Unknown dae kind')

    #
    # Define and compile theano functions
    #
    print "Defining and compiling theano functions"

    # predictions and theano functions

    # fcn prediction
    if segm_net in ['fcn_fcresnet']:
        # no need if we use keras, viva keras!
        pass
    else:
        pred_fcn = lasagne.layers.get_output(fcn, deterministic=True, batch_norm_use_averages=False)
        pred_fcn_fn = theano.function([input_x_var], pred_fcn)

    pred_dae = lasagne.layers.get_output(dae, deterministic=True)
    pred_dae_fn = theano.function(input_concat_h_vars+[y_hat_var], pred_dae)

    # Reshape iterative inference output to b01,c
    y_hat_dimshuffle = y_hat_var.dimshuffle((0, 2, 3, 1))
    sh = y_hat_dimshuffle.shape
    y_hat_2D = y_hat_dimshuffle.reshape((T.prod(sh[:3]), sh[3]))

    # Reshape iterative inference output to b01,c
    target_var_dimshuffle = target_var.dimshuffle((0, 2, 3, 1))
    sh2 = target_var_dimshuffle.shape
    target_var_2D = target_var_dimshuffle.reshape((T.prod(sh2[:3]), sh2[3]))

    # derivative of energy wrt input and theano function
    de = - (pred_dae - y_hat_var)
    de_fn = theano.function(input_concat_h_vars+[y_hat_var], de)

    # metrics and theano functions
    test_loss =  squared_error(y_hat_var, target_var, void)
    test_acc = accuracy(y_hat_2D, target_var_2D, void_labels, one_hot=True)
    test_jacc = jaccard(y_hat_2D, target_var_2D, n_classes, one_hot=True)
    val_fn = theano.function([y_hat_var, target_var], [test_acc, test_jacc, test_loss])

    #
    # Infer
    #
    print 'Start infering'
    rec_tot = 0
    rec_tot_fcn = 0
    rec_tot_dae = 0
    acc_tot = 0
    acc_tot_fcn = 0
    jacc_tot = 0
    jacc_tot_fcn = 0
    acc_tot_dae = 0
    jacc_tot_dae = 0

    valid_mat = np.zeros((2, n_classes, num_iter))

    print 'Inference step: '+str(learn_step)+ 'num iter '+str(num_iter)
    for i in range(n_batches_test):
        info_str = "Batch %d out of %d" % (i+1, n_batches_test)
        print '-'*30
        print '*'*5 + info_str + '*'*5
        print '-'*30

        # Get minibatch
        X_test_batch, L_test_batch = data_iter.next()
        if segm_net in ['fcn_fcresnet']:
            # flip labels to the format used in Keras
            L_test_batch = 1 - L_test_batch
        L_test_batch = L_test_batch.astype(_FLOATX)

        # Compute fcn prediction y and h
        if segm_net in ['fcn_fcresnet']:
            pred_test_batch = fcn.predict(X_test_batch)
        else:
            pred_test_batch = fcn_fn(X_test_batch)
        Y_test_batch = pred_test_batch[-1]
        H_test_batch = pred_test_batch[:-1]

        # Compute metrics before iterative inference
        acc_fcn, jacc_fcn, rec_fcn = val_fn(Y_test_batch, L_test_batch)
        acc_tot_fcn += acc_fcn
        jacc_tot_fcn += jacc_fcn
        rec_tot_fcn += rec_fcn
        Y_test_batch_fcn = Y_test_batch
        print_results('>>>>> FCN:', rec_tot_fcn, acc_tot_fcn, jacc_tot_fcn, i+1)

        # Compute dae output and metrics after dae
        Y_test_batch_dae = pred_dae_fn(*(H_test_batch+[Y_test_batch]))
        acc_dae, jacc_dae, rec_dae = val_fn(Y_test_batch_dae, L_test_batch)
        acc_tot_dae += acc_dae
        jacc_tot_dae += jacc_dae
        rec_tot_dae += rec_dae
        print_results('>>>>> FCN+DAE:', rec_tot_dae, acc_tot_dae, jacc_tot_dae, i+1)

        for im in range(X_test_batch.shape[0]):
            print('-----------------------')
            h_im = [el[np.newaxis, im] for el in H_test_batch]
            y_im = Y_test_batch[np.newaxis, im]
            t_im = L_test_batch[np.newaxis, im]

            # Iterative inference
            for it in range(num_iter):
                # Compute gradient
                grad = de_fn(*(h_im+[y_im]))

                # Update prediction
                y_im = y_im - learn_step * grad

                # Clip prediction
                y_im = np.clip(y_im, 0.0, 1.0)

                norm = np.linalg.norm(grad, axis=1).mean()
                if norm < _EPSILON:
                    break

                acc_iter, jacc_iter, rec_iter = val_fn(y_im, t_im)
                print rec_iter, acc_iter, np.nanmean(jacc_iter[0, :]/jacc_iter[1, :])
                valid_mat[:, :, it] += jacc_iter

    # Print summary of how things went
    print('-------------------------------------------------------------------')
    print('------------------------------SUMMARY------------------------------')
    print('-------------------------------------------------------------------')
    print_results('>>>>> FCN:', rec_tot_fcn, acc_tot_fcn, jacc_tot_fcn, i+1)
    print_results('>>>>> FCN+DAE:', rec_tot_dae, acc_tot_dae, jacc_tot_dae, i+1)

    res = np.nanmean(valid_mat[0, :, :] / valid_mat[1, :, :], axis=0)
    print res.max()
    print res.argmax()
    print learn_step

    print savepath
    np.savez(os.path.join(savepath, 'iterations'+str(learn_step)+'.npz'), valid_mat)

    # Move segmentations
    if savepath != loadpath:
        print('Copying images to {}'.format(loadpath))
        copy_tree(savepath, os.path.join(loadpath, 'img_plots', str(learn_step), which_set))


def main():
    parser = argparse.ArgumentParser(description='Iterative inference.')

    parser.add_argument('-dataset',
                        type=str,
                        default='em',
                        help='Dataset.')
    parser.add_argument('-segmentation_net',
                        type=str,
                        default='fcn_fcresnet',
                        help='Segmentation network.')
    parser.add_argument('-step',
                        type=float,
                        default=0.0005,
                        help='step')
    parser.add_argument('--num_iter',
                        '-ne',
                        type=int,
                        default=50,
                        help='Max number of iterations')
    parser.add_argument('-which_set',
                        type=str,
                        default='val',
                        help='Inference set')
    parser.add_argument('-dae_dict',
                        type=dict,
                        default={'kind': 'standard', 'dropout': 0.2, 'skip': True,
                                  'unpool_type': 'trackind', 'noise':0.1,
                                  'concat_h': ['pool4'], 'from_gt': False,
                                  'n_filters': 64, 'conv_before_pool': 1,
                                  'additional_pool': 2,
                                  'path_weights': '', 'layer': 'probs_dimshuffle',
                                 'exp_name' : 'lmb1_mse_', 'bn': 0},
                        help='DAE kind and parameters')
    parser.add_argument('-training_dict',
                        type=dict,
                        default={'training_loss': ['crossentropy',
                                                   'squared_error'],
                                 'learning_rate': 0.001, 'lr_anneal': 0.99,
                                 'weight_decay':0.0001, 'optimizer': 'rmsprop'},
                        help='Training parameters')
    parser.add_argument('-full_im_ft',
                        type=bool,
                        default=False,
                        help='Whether to finetune at full image resolution')
    parser.add_argument('-ae_h',
                        type=bool,
                        default=False,
                        help='Whether to reconstruct intermediate h')
    parser.add_argument('-data_augmentation',
                        type=bool,
                        default=True,
                        help='Dictionary of data augmentation to be used')
    parser.add_argument('-test_from_0_255',
                        type=bool,
                        default=True,
                        help='Whether to train from images within 0-255 range')

    args = parser.parse_args()

    inference(args.dataset, args.segmentation_net, float(args.step),
              int(args.num_iter), which_set=args.which_set,
              savepath=SAVEPATH, loadpath=LOADPATH,
              test_from_0_255=args.test_from_0_255, ae_h=args.ae_h,
              dae_dict_updates=args.dae_dict, data_augmentation=args.data_augmentation,
              training_dict=args.training_dict, full_im_ft=args.full_im_ft)


if __name__ == "__main__":
    main()
