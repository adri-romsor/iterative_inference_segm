from lasagne.layers import InputLayer, DropoutLayer, GaussianNoiseLayer
from lasagne.layers import Pool2DLayer as PoolLayer, ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer


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
    else:
        out = in1

    return pos, out


def buildFCN_down(input_var, concat_vars,
                  n_classes=21, concat_layers=['pool5']):

    '''
    Build fcn contracting path
    '''

    assert all([el in ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6',
                       'fc7', 'input'] for el in concat_layers])

    net = {}
    pos = 0

    # Contracting path
    net['input'] = InputLayer((None, n_classes, None, None),
                              input_var)

    # Noise
    net['noisy_input'] = GaussianNoiseLayer(net['input'])

    pos, out = concatenate(net, 'noisy_input', concat_layers, concat_vars, pos)

    if concat_layers[-1] == 'noisy_input':
        return net

    # pool 1
    net['conv1_1'] = ConvLayer(
        net[out],
        64, 3, pad=100, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad='same', flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)

    pos, out = concatenate(net, 'pool1', concat_layers, concat_vars, pos)
    if concat_layers[-1] == 'pool1':
        return net

    # pool 2
    net['conv2_1'] = ConvLayer(
        net[out], 128, 3, pad='same', flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad='same', flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)

    pos, out = concatenate(net, 'pool2', concat_layers, concat_vars, pos)
    if concat_layers[-1] == 'pool2':
        return net

    # pool 3
    net['conv3_1'] = ConvLayer(
        net[out], 256, 3, pad='same', flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad='same', flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad='same', flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)

    pos, out = concatenate(net, 'pool3', concat_layers, concat_vars, pos)
    if concat_layers[-1] == 'pool3':
        return net

    # pool 4
    net['conv4_1'] = ConvLayer(
        net[out], 512, 3, pad='same', flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad='same', flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad='same', flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)

    pos, out = concatenate(net, 'pool4', concat_layers, concat_vars, pos)
    if concat_layers[-1] == 'pool4':
        return net

    # pool 5
    net['conv5_1'] = ConvLayer(
        net[out], 512, 3, pad='same', flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad='same', flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad='same', flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    pos, out = concatenate(net, 'pool5', concat_layers, concat_vars, pos)
    if concat_layers[-1] == 'pool5':
        return net

    # fc6
    net['fc6'] = ConvLayer(
        net[out], 4096, 7, pad='valid', flip_filters=False)
    net['fc6_dropout'] = DropoutLayer(net['fc6'])

    pos, out = concatenate(net, 'fc6_dropout',
                                concat_layers, concat_vars, pos)
    if concat_layers[-1] == 'fc6':
        return net

    # fc7
    net['fc7'] = ConvLayer(
        net[out], 4096, 1, pad='valid', flip_filters=False)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)

    pos, out = concatenate(net, 'fc7_dropout',
                                concat_layers, concat_vars, pos)
    if concat_layers[-1] == 'fc7':
        return net
