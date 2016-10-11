import numpy as np
import os
import lasagne

import model_helpers

import theano.tensor as T

from lasagne.nonlinearities import linear, rectify
from lasagne.layers import (DilatedConv2DLayer, Conv2DLayer,
                            ConcatLayer, InputLayer, PadLayer, DimshuffleLayer,
                            ReshapeLayer, NonlinearityLayer)
from layers.mylayers import GaussianNoiseLayerSoftmax
from lasagne.init import Initializer


def concatenate(net, in1, concat_layers, concat_vars, pos):
    if concat_layers[pos] == 'input':
        concat_layers[pos] = in1

    if in1 in concat_layers:
        net[in1 + '_h'] = InputLayer((None, net[in1].input_shape[1] if
                                      'input' not in concat_layers[pos]
                                      else 3, None, None), concat_vars[pos])
        net[in1 + '_concat'] = ConcatLayer((net[in1 + '_h'],
                                            net[in1]), axis=1, cropping=None)
        pos += 1
        out = in1 + '_concat'

        laySize = net[out].output_shape
        n_cl = laySize[1]
        print('Number of feature maps (concat):', n_cl)
    else:
        out = in1

    if concat_layers[pos-1] == 'noisy_input':
        concat_layers[pos-1] = 'input'

    return pos, out


def buildDAE_contextmod(concat_vars, input_mask_var, n_classes,
                        concat_layers=['input'], noise=0.1,
                        path_weights='/Tmp/romerosa/itinf/models/',
                        model_name='dae_model.npz',
                        trainable=False, load_weights=False,
                        out_nonlin=linear):

    '''
    Build score model as a context module
    '''
    assert all([el in ['input'] for el in concat_layers])
    net = {}
    pos = 0

    # Contracting path
    net['input'] = InputLayer((None, n_classes, None, None), input_mask_var)

    # Noise
    if noise > 0:
        net['noisy_input'] = GaussianNoiseLayerSoftmax(net['input'],
                                                       sigma=noise)
        in_next = 'noisy_input'
    else:
        in_next = 'input'

    pos, out = concatenate(net, in_next, concat_layers, concat_vars, pos)

    class IdentityInit(Initializer):
        """ We adapt the same initializiation method than in the paper"""

        def sample(self, shape):
            n_filters, n_filters2, filter_size, filter_size2 = shape
            assert ((n_filters == n_filters2) & (filter_size == filter_size2))
            assert (filter_size % 2 == 1)

            W = np.zeros(shape, dtype='float32')
            for i in range(n_filters):
                W[i, i, filter_size / 2, filter_size / 2] = 1.
            return W

    net['conv1'] = Conv2DLayer(net[out], n_classes, 3,
                               pad='same', nonlinearity=rectify,
                               flip_filters=False)
    net['pad1'] = PadLayer(net['conv1'], width=32, val=0, batch_ndim=2)
    net['dilconv1'] = DilatedConv2DLayer(net['pad1'],
                                         n_classes, 3, 1,
                                         W=IdentityInit(),
                                         nonlinearity=rectify)
    net['dilconv2'] = DilatedConv2DLayer(net['dilconv1'],
                                         n_classes, 3, 2,
                                         W=IdentityInit(),
                                         nonlinearity=rectify)
    net['dilconv3'] = DilatedConv2DLayer(net['dilconv2'],
                                         n_classes, 3, 4,
                                         W=IdentityInit(),
                                         nonlinearity=rectify)
    net['dilconv4'] = DilatedConv2DLayer(net['dilconv3'],
                                         n_classes, 3, 8,
                                         W=IdentityInit(),
                                         nonlinearity=rectify)
    net['dilconv5'] = DilatedConv2DLayer(net['dilconv4'],
                                         n_classes, 3, 16,
                                         W=IdentityInit(),
                                         nonlinearity=rectify)
    net['dilconv6'] = DilatedConv2DLayer(net['dilconv5'],
                                         n_classes, 3, 1,
                                         W=IdentityInit(),
                                         nonlinearity=rectify)
    net['dilconv7'] = DilatedConv2DLayer(net['dilconv6'],
                                         n_classes, 1, 1,
                                         W=IdentityInit(),
                                         nonlinearity=linear)

    # Final dimshuffle, reshape and softmax
    net['final_dimshuffle'] = DimshuffleLayer(net['dilconv7'],
                                              (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = ReshapeLayer(net['final_dimshuffle'],
                                        (T.prod(laySize[0:3]),
                                         laySize[3]))
    net['probs'] = NonlinearityLayer(net['final_reshape'],
                                     nonlinearity=out_nonlin)

    # Go back to 4D
    net['probs_reshape'] = ReshapeLayer(net['probs'], (laySize[0], laySize[1],
                                                       laySize[2], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'],
                                              (0, 3, 1, 2))
    # print('Input to last layer: ', net['probs_dimshuffle'].input_shape)
    print(net.keys())

    # Load weights
    if load_weights:
        with np.load(os.path.join(path_weights, model_name)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]

        lasagne.layers.set_all_param_values(net['probs_dimshuffle'],
                                            param_values)

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(net['probs_dimshuffle'], single=False)

    return net['probs_dimshuffle']
