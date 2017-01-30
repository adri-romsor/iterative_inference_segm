import pickle

import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import InputLayer, Pool2DLayer as PoolLayer

from layers.mylayers import RGBtoBGRLayer

import model_helpers

# Path to pretrained models
pathPretrained = '/data/lisatmp3/romerosa/pretrained/'


def build_VGG16(inputSize, input_var=None,
                pathVGG16=pathPretrained+'vgg16.pkl',
                last_layer='pool5', trainable=False):
    """
    Construct VGG16 convnet
    """

    net = {}
    net['input'] = InputLayer((None, 3, None, None), input_var)
    net['bgr'] = RGBtoBGRLayer(net['input'])
    net['conv1_1'] = ConvLayer(net['bgr'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3,
                               pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3,
                               pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3,
                               pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3,
                               pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3,
                               pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3,
                               pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3,
                               pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3,
                               pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3,
                               pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    pretrained_values = pickle.load(open(pathVGG16))['param values']

    nlayers = len(lasagne.layers.get_all_params(net[last_layer]))

    lasagne.layers.set_all_param_values(net[last_layer],
                                        pretrained_values[:nlayers])

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(net)

    return net[last_layer]
