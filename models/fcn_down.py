from lasagne.layers import InputLayer, GaussianNoiseLayer
from lasagne.layers import Pool2DLayer as PoolLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer, DropoutLayer


def concatenate(net, in1, concat_layers, concat_vars, pos):
    if concat_layers[pos] == 'input':
        concat_layers[pos] = 'noisy_input'

    if in1 in concat_layers:
        net[in1 + '_h'] = InputLayer((None, net[in1].input_shape[1] if
                                      concat_layers[pos] != 'noisy_input' else 3,
                                      None, None), concat_vars[pos])
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


def buildFCN_down(input_var, concat_vars,
                  n_classes=21, concat_layers=['pool5'], noise=0.1,
                  n_filters=64, conv_before_pool=1, additional_pool=0,
                  dropout=0.):

    '''
    Build fcn contracting path
    '''

    assert all([el in ['pool1', 'pool2', 'pool3', 'pool4', 'input']
                for el in concat_layers])

    if 'pool' in concat_layers[-1]:
        n_pool = int(concat_layers[-1][-1])
    else:
        n_pool = 0

    net = {}
    pos = 0

    # Contracting path
    net['input'] = InputLayer((None, n_classes, None, None),
                              input_var)

    # Noise
    net['noisy_input'] = GaussianNoiseLayer(net['input'], sigma=noise)

    pos, out = concatenate(net, 'noisy_input', concat_layers, concat_vars, pos)

    if concat_layers[-1] == 'input' and additional_pool == 0:
        return net, out

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

            # Define conv
            if p < 6:
                filters_conv = n_filters*(2**p)

            net['conv'+str(p+1)+'_'+str(i)] = ConvLayer(
                net[out], filters_conv, 3,
                pad=pad_type, flip_filters=False)
            out = 'conv'+str(p+1)+'_'+str(i)

            if p > n_pool and dropout > 0.:
                net[out+'_drop'] = DropoutLayer(net[out], p=dropout)
                out += '_drop'

        # Define pool
        net['pool'+str(p+1)] = PoolLayer(net['conv'+str(p+1)+'_'+str(i)], 2)
        out = 'pool'+str(p+1)
        laySize = net['pool'+str(p+1)].input_shape
        n_cl = laySize[1]
        print('Number of feature maps (out):', n_cl)

        if p < n_pool:
            pos, out = concatenate(net, 'pool'+str(p+1), concat_layers,
                                   concat_vars, pos)

        last_layer = out

    return net, last_layer
