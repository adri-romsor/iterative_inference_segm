import numpy as np
import lasagne
from lasagne.layers import InputLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import linear, sigmoid

import model_helpers
from fcn_up import buildFCN_up
from fcn_down import buildFCN_down


def buildDAE(input_repr_var, input_mask_var, n_classes,
             layer_h='input', filter_size=[], kernel_size=[],
             void_labels=[],
             path_weights='/Tmp/romerosa/itinf/models/' +
             'camvid/dae_model.npz',
             trainable=False, load_weights=False):

    '''
    Build score model
    '''

    # Compute number of output classes of the DAE and input sizes
    n_classes = n_classes + (1 if void_labels else 0)

    # Build fcn to extract representation from y
    fcn_down = buildFCN_down(input_mask_var, n_classes=n_classes)

    dae = fcn_down

    # Number of representation channels
    if layer_h == 'input':
        layer_h = 'noisy_input'
        nb_repr_channels = 3
    else:
        nb_repr_channels = fcn_down[layer_h].input_shape[1]

    input_repr_size = (None, nb_repr_channels, 17, 21)

    # 2 input layers to the energy function (h, y)
    dae['input_repr'] = InputLayer(input_repr_size,
                                   input_repr_var)

    # Concatenate the 2 inputs (h, h_y)
    dae['concat'] = ConcatLayer((dae['input_repr'], fcn_down[layer_h]),
                                axis=1, cropping=None)

    # Stack encoder on top of concatenation
    l_enc = 0
    for f, k in zip(filter_size, kernel_size):
        dae['encoder' + str(l_enc)] = ConvLayer(dae['concat' if l_enc == 0 else
                                                'encoder' + str(l_enc-1)],
                                                f, k, flip_filters=False,
                                                pad='same',
                                                nonlinearity=sigmoid)
        l_enc += 1

    # Count the number of pooling layers to know how many upsamplings we
    # should perform
    unpool = np.sum([isinstance(el, PoolLayer) for el in
                     lasagne.layers.get_all_layers(dae['concat'])])

    # Stack decoder
    filter_size[0] = n_classes if unpool == 0 else nb_repr_channels
    l_dec = 0
    for f, k in list(reversed(zip(filter_size, kernel_size))):
        dae['decoder' + str(l_dec)] = \
            ConvLayer(dae['encoder'+str(l_enc-1) if l_dec == 0 else
                          'decoder' + str(l_dec-1)],
                      f, k, flip_filters=False, pad='same',
                      nonlinearity=linear)
        l_dec += 1

    # Unpooling
    fcn_up = buildFCN_up(dae, 'decoder'+str(l_dec-1), unpool,
                         n_classes=n_classes)

    dae.update(fcn_up)

    # Load weights
    if load_weights:
        with np.load(path_weights) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(dae['probs_dimshuffle'], param_values)

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(dae['probs_dimshuffle'], single=False)

    return dae['probs_dimshuffle']
