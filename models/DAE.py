import numpy as np
import lasagne
from lasagne.layers import InputLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import linear, sigmoid

import model_helpers


def buildDAE(input_repr_var, input_mask_var, nb_in_channels, n_classes,
             filter_size=[], kernel_size=[],
             path_weights='/Tmp/romerosa/itinf/models/' +
             'camvid/dae_model.npz',
             trainable=False, load_weights=False):

    '''
    Build score model
    '''

    net = {}

    # Input
    net['input_repr'] = InputLayer((None, nb_in_channels, None, None),
                                   input_repr_var)
    net['input_mask'] = InputLayer((None, n_classes, None, None),
                                   input_mask_var)
    # Noise layer
    net['noisy_mask'] = lasagne.layers.GaussianNoiseLayer(net['input_mask'])

    net = ConcatLayer((net['input_repr'], net['noisy_mask']),
                      axis=1,
                      cropping=None)
    # Encoder
    for f, k in zip(filter_size, kernel_size):
        net = ConvLayer(net, f, k, flip_filters=False, pad='same',
                        nonlinearity=sigmoid)

    # Decoder
    filter_size[0] = n_classes
    for f, k in list(reversed(zip(filter_size, kernel_size))):
        net = ConvLayer(net, f, k, flip_filters=False, pad='same',
                        nonlinearity=linear)

    # Load weights
    if load_weights:
        with np.load(path_weights) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(net, param_values)

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(net)

    return net
