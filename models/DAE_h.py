import numpy as np
import os
import lasagne

import model_helpers
from fcn_up import buildFCN_up
from fcn_down import buildFCN_down

from lasagne.nonlinearities import linear


def buildDAE(input_concat_h_vars, input_mask_var, n_classes, nb_features_to_concat,
             padding, ae_h=False, void_labels=[], path_weights='/Tmp/romerosa/itinf/models/',
             model_name='dae_model.npz', trainable=False, load_weights=False,
             out_nonlin=linear, concat_h=['input'], noise=0.1, n_filters=64,
             conv_before_pool=1, additional_pool=0, dropout=0., skip=False,
             unpool_type='standard'):
    '''
    Build score model
    '''

    # Build fcn contracting path
    fcn_down, last_layer_down = buildFCN_down(
        input_mask_var, input_concat_h_vars,
        nb_features_to_concat=nb_features_to_concat,
        padding=padding,
        n_classes=n_classes,
        concat_layers=concat_h,
        noise=noise, n_filters=n_filters,
        conv_before_pool=conv_before_pool,
        additional_pool=additional_pool, ae_h=ae_h)

    dae = fcn_down

    # Count the number of pooling layers to know how many upsamplings we
    # should perform
    if 'pool' in concat_h[-1]:
        n_pool = int(concat_h[-1][-1]) + additional_pool
    else:
        n_pool = additional_pool

    # Unpooling
    fcn_up = buildFCN_up(dae, last_layer_down, n_pool, skip=skip,
                         n_classes=n_classes, unpool_type=unpool_type,
                         out_nonlin=out_nonlin, ae_h=ae_h,
                         additional_pool=additional_pool)

    dae.update(fcn_up)

    # Load weights
    if load_weights:
        with np.load(os.path.join(path_weights, model_name)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(dae['probs_dimshuffle'],
                                            param_values)

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(dae['probs_dimshuffle'], single=False)

    return dae['probs_dimshuffle']
