import numpy as np

import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer, \
    NonlinearityLayer, DimshuffleLayer, ConcatLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.nonlinearities import softmax, linear

from model_helpers import freezeParameters


def buildFCN8(nb_in_channels, input_var,
              path_weights='/Tmp/romerosa/itinf/models/' +
              'camvid/fcn8_model.npz',
              n_classes=21, load_weights=True,
              void_labels=[], trainable=False,
              layer=['probs_dimshuffle']):

    '''
    Build fcn8 model (generator)
    '''

    net = {}

    # Contracting path
    net['input'] = InputLayer((None, nb_in_channels, None, None),
                              input_var)

    # pool 1
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=100, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad='same', flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)

    # pool 2
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad='same', flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad='same', flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)

    # pool 3
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad='same', flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad='same', flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad='same', flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)

    # pool 4
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad='same', flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad='same', flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad='same', flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)

    # pool 5
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad='same', flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad='same', flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad='same', flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    # fc6
    net['fc6'] = ConvLayer(
        net['pool5'], 4096, 7, pad='valid', flip_filters=False)
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
    net['final_dimshuffle'] = DimshuffleLayer(net['score'], (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = ReshapeLayer(net['final_dimshuffle'],
                                        (T.prod(laySize[0:3]),
                                         laySize[3]))
    net['probs'] = NonlinearityLayer(net['final_reshape'],
                                     nonlinearity=softmax)

    # Load weights
    if load_weights:
        with np.load(path_weights) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net['probs'], param_values)

    if not trainable:
        freezeParameters(net['probs'], single=False)

    if any(void_labels):
        layVoid = lasagne.layers.get_output(net['probs']).shape
        input_discrim_var = T.zeros((layVoid[0], 1))
        net['input_void'] = InputLayer((None, 1), input_discrim_var)
        net['concat'] = ConcatLayer([net['probs'], net['input_void']],
                                    axis=1, cropping=None)
        n_classes = n_classes + 1
    else:
        net['concat'] = net['probs']

    # Go back to 4D
    net['probs_reshape'] = ReshapeLayer(net['concat'], (laySize[0], laySize[1],
                                                        laySize[2], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'],
                                              (0, 3, 1, 2))

    return [net[el] for el in layer]
