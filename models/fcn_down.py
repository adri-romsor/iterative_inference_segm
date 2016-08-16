import lasagne
from lasagne.layers import InputLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer


def buildFCN_down(input_var, n_classes=21, layer='pool5'):

    '''
    Build fcn contracting path
    '''

    assert layer in ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6',
                     'fc7', 'input']

    net = {}

    # Contracting path
    net['input'] = InputLayer((None, n_classes, None, None),
                              input_var)

    # Noise
    net['noisy_input'] = lasagne.layers.GaussianNoiseLayer(net['input'])

    if layer == 'input':
        return

    # pool 1
    net['conv1_1'] = ConvLayer(
        net['noisy_input'], 64, 3, pad=100, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad='same', flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)

    if layer == 'pool1':
        return net

    # pool 2
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad='same', flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad='same', flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)

    if layer == 'pool2':
        return net

    # pool 3
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad='same', flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad='same', flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad='same', flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)

    if layer == 'pool3':
        return net

    # pool 4
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad='same', flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad='same', flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad='same', flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)

    if layer == 'pool4':
        return net

    # pool 5
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad='same', flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad='same', flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad='same', flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    if layer == 'pool5':
        return net

    # fc6
    net['fc6'] = ConvLayer(
        net['pool5'], 4096, 7, pad='valid', flip_filters=False)
    net['fc6_dropout'] = DropoutLayer(net['fc6'])

    if layer == 'fc6':
        return net

    # fc7
    net['fc7'] = ConvLayer(
        net['fc6_dropout'], 4096, 1, pad='valid', flip_filters=False)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)

    if layer == 'fc7':
        return net
