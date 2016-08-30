import numpy as np
import os
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import linear, sigmoid

import model_helpers
from fcn_up import buildFCN_up
from fcn_down import buildFCN_down


def buildDAE(input_repr_var, input_mask_var, n_classes,
             layer_h=['input'], filter_size=[], kernel_size=[],
             void_labels=[], skip=False, unpool_type='standard',
             path_weights='/Tmp/romerosa/itinf/models/',
             model_name='dae_model.npz',
             trainable=False, load_weights=False):

    '''
    Build score model
    '''

    # Build fcn to extract representation from y
    fcn_down = buildFCN_down(input_mask_var, input_repr_var,
                             n_classes=n_classes, concat_layers=layer_h)

    dae = fcn_down

    # Stack encoder on top of concatenation
    l_enc = 0
    for f, k in zip(filter_size, kernel_size):
        dae['encoder' + str(l_enc)] = \
            ConvLayer(dae[layer_h[-1]+'_concat' if l_enc == 0 else
                          'encoder' + str(l_enc-1)],
                      f, k, flip_filters=False, pad='same',
                      nonlinearity=sigmoid)
        l_enc += 1

    # Count the number of pooling layers to know how many upsamplings we
    # should perform
    unpool = np.sum([isinstance(el, PoolLayer) for el in
                     lasagne.layers.get_all_layers(dae[layer_h[-1] +
                                                       '_concat'])])

    # Stack decoder
    filter_size[0] = n_classes if unpool == 0 else \
        fcn_down[layer_h[-1]].input_shape[1]
    l_dec = 0
    for f, k in list(reversed(zip(filter_size, kernel_size))):
        dae['decoder' + str(l_dec)] = \
            ConvLayer(dae['encoder'+str(l_enc-1) if l_dec == 0 else
                          'decoder' + str(l_dec-1)],
                      f, k, flip_filters=False, pad='same',
                      nonlinearity=linear)
        l_dec += 1

    # Unpooling
    fcn_up = buildFCN_up(dae, 'decoder'+str(l_dec-1), unpool, skip=skip,
                         n_classes=n_classes, unpool_type=unpool_type)

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
