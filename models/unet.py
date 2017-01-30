import numpy as np

import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, ConcatLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax
from lasagne.layers import Deconv2DLayer as DeconvLayer
from layers.MIlayers import CropLayer
import model_helpers


def build_unet(inputSize, dropout, input_var=None,
               path_weights="/data/lisatmp4/erraqabi/results/Unet/" +
               "polyp_unet_drop_1e-5_750_epoch/" +
               "u_net_model.npz",
               nclasses=2, trainable=False,
               last_pretrained_layer="probs",
               temperature=1):
    """
    Build u-net model
    """

    print "Building pretrained model"
    net = {}
    net['input'] = InputLayer((None, inputSize[0], inputSize[1], inputSize[2]),
                              input_var)
    net['conv1_1'] = ConvLayer(net['input'], 64, 3)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3)

    net['pool1'] = PoolLayer(net['conv1_2'], 2, ignore_border=False)

    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3)

    net['pool2'] = PoolLayer(net['conv2_2'], 2, ignore_border=False)

    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3)

    net['pool3'] = PoolLayer(net['conv3_2'], 2, ignore_border=False)

    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3)

    if dropout:
        net['drop1'] = DropoutLayer(net['conv4_2'])
        prev_layer1 = 'drop1'
    else:
        prev_layer1 = 'conv4_2'

    net['pool4'] = PoolLayer(net[prev_layer1], 2, ignore_border=False)

    net['conv5_1'] = ConvLayer(net['pool4'], 1024, 3)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 1024, 3,  flip_filters=False)

    if dropout:
        net['drop2'] = DropoutLayer(net['conv5_2'])
        prev_layer2 = 'drop2'
    else:
        prev_layer2 = 'conv5_2'

    net['upconv4'] = DeconvLayer(net[prev_layer2], 512, 2, stride=2)
    net['Concat_4'] = ConcatLayer(
        (net['conv4_2'], net['upconv4']), axis=1,
        cropping=[None, None, 'center', 'center'])

    net['conv6_1'] = ConvLayer(net['Concat_4'], 512, 3)
    net['conv6_2'] = ConvLayer(net['conv6_1'], 512, 3)

    net['upconv3'] = DeconvLayer(net['conv6_2'], 256, 2, stride=2)
    net['Concat_3'] = ConcatLayer(
        (net['conv3_2'], net['upconv3']), axis=1,
        cropping=[None, None, 'center', 'center'])

    net['conv7_1'] = ConvLayer(net['Concat_3'], 256, 3)
    net['conv7_2'] = ConvLayer(net['conv7_1'], 256, 3)

    net['upconv2'] = DeconvLayer(net['conv7_2'], 128, 2, stride=2)
    net['Concat_2'] = ConcatLayer(
        (net['conv2_2'], net['upconv2']), axis=1,
        cropping=[None, None, 'center', 'center'])

    net['conv8_1'] = ConvLayer(net['Concat_2'], 128, 3)
    net['conv8_2'] = ConvLayer(net['conv8_1'], 128, 3)

    net['upconv1'] = DeconvLayer(net['conv8_2'], 64, 2, stride=2)
    net['Concat_1'] = ConcatLayer(
        (net['conv1_2'], net['upconv1']), axis=1,
        cropping=[None, None, 'center', 'center'])

    net['conv9_1'] = ConvLayer(net['Concat_1'], 64, 3)
    net['conv9_2'] = ConvLayer(net['conv9_1'], 64, 3)

    net['conv10'] = ConvLayer(net['conv9_2'], nclasses, 1,
                              nonlinearity=lasagne.nonlinearities.identity)

    laySize = lasagne.layers.get_output_shape(net['conv10'])
    # laySize = lasagne.layers.get_output(net['conv10']).shape
    net['final_crop'] = CropLayer(
        net['conv10'], np.asarray(laySize[2:])-np.asarray(inputSize[1:]) + 184,
        centered=False)

    net['final_dimshuffle'] = \
        lasagne.layers.DimshuffleLayer(net['final_crop'], (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = \
        lasagne.layers.ReshapeLayer(net['final_dimshuffle'],
                                    (T.prod(laySize[0:3]),
                                     laySize[3]))
    net['probs'] = lasagne.layers.NonlinearityLayer(net['final_reshape'],
                                                    nonlinearity=softmax)

    with np.load(path_weights) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    if last_pretrained_layer == "probs":
        param_values[-1] /= temperature
        param_values[-2] /= temperature

    nlayers = len(lasagne.layers.get_all_params(net[last_pretrained_layer]))

    lasagne.layers.set_all_param_values(net[last_pretrained_layer],
                                        param_values[:nlayers])

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(net)

    return net[last_pretrained_layer]


if __name__ == '__main__':
    network = build_unet([3, 250, 500], dropout=True,
                         input_var=T.tensor4('inputs_var'))
