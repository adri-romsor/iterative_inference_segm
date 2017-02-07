from lasagne.layers import InputLayer, GaussianNoiseLayer
from lasagne.layers import Pool2DLayer as PoolLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer, DropoutLayer
from layers.mylayers import GaussianNoiseLayerSoftmax

import model_helpers


def buildFCN_down(input_var, concat_h_vars,
                  n_classes=21, concat_layers=['pool5'], noise=0.1,
                  n_filters=64, conv_before_pool=1, additional_pool=0,
                  dropout=0.):

    '''
    Build fcn contracting path. The contracting path is built by combining
    convolution and pooling layers, at least until the last concatenation is
    reached. The last concatenation can be eventually followed by extra
    convolution and pooling layers.

    Parameters
    ----------
    input_var: theano tensor, input of the network
    concat_h_vars: list of theano tensors, intermediate inputs of the network
    n_classes: int, number of classes
    concat_layers: list intermediate layers names (layers we want to
        concatenate)
    noise: float, noise
    n_filters: int, number of filters of each convolution (increases every time
        resolution is downsampled)
    conv_before_pool: int, number of convolutions before a pooling layer
    additional_pool: int, number of poolings following the concatenation of the
        last layer layer in `concat_h_vars` and `concat_layers`
    dropout: float, dropout probability
    '''

    assert all([el in ['pool1', 'pool2', 'pool3', 'pool4', 'input']
                for el in concat_layers])

    if 'pool' in concat_layers[-1]:
        n_pool = int(concat_layers[-1][-1])
    else:
        n_pool = 0

    net = {}
    pos = 0

    #
    # Contracting path
    #

    # input
    net['input'] = InputLayer((None, n_classes, None, None),
                              input_var)

    # noise
    if noise > 0:
        # net['noisy_input'] = GaussianNoiseLayerSoftmax(net['input'],
        #                                                sigma=noise)
        net['noisy_input'] = GaussianNoiseLayer(net['input'],
                                                sigma=noise)
        in_next = 'noisy_input'
    else:
        in_next = 'input'

    # check whether we need to concatenate concat_h
    pos, out = model_helpers.concatenate(net, in_next, concat_layers,
                                         concat_h_vars, pos)

    if concat_layers[-1] == 'input' and additional_pool == 0:
        raise ValueError('It seems your DAE will have no conv/pooling layers!')

    # start building the network
    for p in range(n_pool+additional_pool):
        # add conv + pool
        for i in range(1, conv_before_pool+1):
            # Choose padding type: this is defined according to the
            # layers:
            # - if have several concats, we padding 100 in the first
            # layer for fcn8 compatibility
            # - if concatenation is only performed at the input, we
            # don't pad
            if p == 0 and i == 1 and len(concat_layers) == 1 and \
               concat_layers[-1] != 'input':
                pad_type = 100
            else:
                pad_type = 'same'

            # Define conv (follow vgg scheme, limited to 6 due to memory
            # constraints)
            if p < 6:
                filters_conv = n_filters*(2**p)

            # add conv layer
            net['conv'+str(p+1)+'_'+str(i)] = ConvLayer(
                net[out], filters_conv, 3,
                pad=pad_type, flip_filters=False)
            out = 'conv'+str(p+1)+'_'+str(i)

            # add dropout layer
            if p > n_pool and dropout > 0.:
                net[out+'_drop'] = DropoutLayer(net[out], p=dropout)
                out += '_drop'

        # add pooling layer
        net['pool'+str(p+1)] = PoolLayer(net['conv'+str(p+1)+'_'+str(i)], 2)
        out = 'pool'+str(p+1)
        laySize = net['pool'+str(p+1)].input_shape
        n_cl = laySize[1]
        print('Number of feature maps (out):', n_cl)

        # check whether concatenation is required
        if p < n_pool:
            pos, out = model_helpers.concatenate(net, 'pool'+str(p+1),
                                                 concat_layers, concat_h_vars,
                                                 pos)

        last_layer = out

    return net, last_layer
