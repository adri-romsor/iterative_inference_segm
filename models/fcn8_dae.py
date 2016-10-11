import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer,\
    DimshuffleLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import ConcatLayer
from lasagne.nonlinearities import softmax, linear

import model_helpers
from layers.mylayers import GaussianNoiseLayerSoftmax


def buildFCN8_DAE(nb_in_channels, input_var,
                  path_weights='/Tmp/romerosa/itinf/models/' +
                  'camvid/fcn8_model.npz',
                  n_classes=21, load_weights=False,
                  void_labels=[], trainable=False,
                  concat_layers=[], noise=0.1, concat_vars=[],
                  pretrained=False):
    '''
    Build fcn8 model as DAE
    '''

    net = {}
    pos = 0

    assert all([el in ['pool1', 'pool2', 'pool3', 'pool4', 'input']
                for el in concat_layers])

    # Contracting path
    net['input'] = InputLayer((None, nb_in_channels, None, None),
                              input_var)
    # Add noise
    # Noise
    if noise > 0:
        net['noisy_input'] = GaussianNoiseLayerSoftmax(net['input'],
                                                       sigma=noise)
        in_next = 'noisy_input'
    else:
        in_next = 'input'

    pos, out = model_helpers.concatenate(net, in_next, concat_layers,
                                         concat_vars, pos)

    # pool 1
    net['conv1_1'] = ConvLayer(
        net[out], 64, 3, pad=100, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad='same', flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)

    pos, out = model_helpers.concatenate(net, 'pool1', concat_layers,
                                         concat_vars, pos)

    # pool 2
    net['conv2_1'] = ConvLayer(
        net[out], 128, 3, pad='same', flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad='same', flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)

    pos, out = model_helpers.concatenate(net, 'pool2', concat_layers,
                                         concat_vars, pos)

    # pool 3
    net['conv3_1'] = ConvLayer(
        net[out], 256, 3, pad='same', flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad='same', flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad='same', flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)

    pos, out = model_helpers.concatenate(net, 'pool3', concat_layers,
                                         concat_vars, pos)

    # pool 4
    net['conv4_1'] = ConvLayer(
        net[out], 512, 3, pad='same', flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad='same', flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad='same', flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)

    pos, out = model_helpers.concatenate(net, 'pool4', concat_layers,
                                         concat_vars, pos)

    # pool 5
    net['conv5_1'] = ConvLayer(
        net[out], 512, 3, pad='same', flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad='same', flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad='same', flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    pos, out = model_helpers.concatenate(net, 'pool5', concat_layers,
                                         concat_vars, pos)

    # fc6
    net['fc6'] = ConvLayer(
        net[out], 4096, 7, pad='valid', flip_filters=False)
    net['fc6_dropout'] = DropoutLayer(net['fc6'])

    # fc7
    net['fc7'] = ConvLayer(
        net['fc6_dropout'], 4096, 1, pad='valid', flip_filters=False)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)

    net['score_fr'] = ConvLayer(
        net['fc7_dropout'], n_classes, 1, pad='valid', flip_filters=False)

    # Upsampling path

    # Unpool
    net['score2'] = DeconvLayer(net['score_fr'], n_classes, 4, stride=2,
                                crop='valid', nonlinearity=linear)
    net['score_pool4'] = ConvLayer(net['pool4'], n_classes, 1,
                                   pad='same')
    net['score_fused'] = ElemwiseSumLayer((net['score2'],
                                           net['score_pool4']),
                                          cropping=[None, None, 'center',
                                                    'center'])

    # Unpool
    net['score4'] = DeconvLayer(net['score_fused'], n_classes, 4,
                                stride=2, crop='valid', nonlinearity=linear)
    net['score_pool3'] = ConvLayer(net['pool3'], n_classes, 1,
                                   pad='valid')
    net['score_final'] = ElemwiseSumLayer((net['score4'],
                                           net['score_pool3']),
                                          cropping=[None, None, 'center',
                                                    'center'])
    # Unpool
    net['upsample'] = DeconvLayer(net['score_final'], n_classes, 16,
                                  stride=8, crop='valid', nonlinearity=linear)
    upsample_shape = lasagne.layers.get_output_shape(net['upsample'])[1]
    net['input_tmp'] = InputLayer((None, upsample_shape, None,
                                   None), input_var)

    net['score'] = ElemwiseMergeLayer((net['input_tmp'], net['upsample']),
                                      merge_function=lambda input, deconv:
                                      deconv,
                                      cropping=[None, None, 'center',
                                                'center'])

    # Final dimshuffle, reshape and softmax
    net['final_dimshuffle'] = \
        lasagne.layers.DimshuffleLayer(net['score'], (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = \
        lasagne.layers.ReshapeLayer(net['final_dimshuffle'],
                                    (T.prod(laySize[0:3]),
                                     laySize[3]))
    net['probs'] = lasagne.layers.NonlinearityLayer(net['final_reshape'],
                                                    nonlinearity=softmax)

    # Load weights
    if load_weights:
        with np.load(path_weights) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net['probs'], param_values)

    if pretrained:  # be careful, only works camvid!!
        with np.load('/data/lisatmp4/romerosa/itinf/models/camvid/fcn8_model.npz') as f:
            start = 0 if nb_in_channels == f['arr_%d' % 0].shape[1] \
                else 2
            param_values = [f['arr_%d' % i] for i in range(start,
                                                           len(f.files))]
        all_layers = lasagne.layers.get_all_layers(net['probs'])[4:]
        count = 0
        for layer in all_layers:
            layer_params = layer.get_params()
            for p in layer_params:
                try:
                    if hasattr(layer, 'input_layer') and not isinstance(layer.input_layer, ConcatLayer):
                        p.set_value(param_values[count])
                    count += 1
                except KeyError:
                    pass

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(net['probs'])

    # Go back to 4D
    net['probs_reshape'] = ReshapeLayer(net['probs'], (laySize[0], laySize[1],
                                                       laySize[2], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'],
                                              (0, 3, 1, 2))

    return net['probs_dimshuffle']
