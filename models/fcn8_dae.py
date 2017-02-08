import os
import scipy.io as sio
import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DropoutLayer, ReshapeLayer,\
    DimshuffleLayer, GaussianNoiseLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import ConcatLayer
from lasagne.nonlinearities import softmax, linear

import model_helpers
from layers.mylayers import GaussianNoiseLayerSoftmax


def buildFCN8_DAE(input_concat_h_vars, input_mask_var, n_classes, nb_in_channels=3,
                  path_weights='/Tmp/romerosa/itinf/models/',
                  model_name='fcn8_model.npz', trainable=False,
                  load_weights=False, pretrained=False, freeze=False,
                  pretrained_path='/data/lisatmp4/romerosa/itinf/models/camvid/',
                  pascal=False, return_layer='probs_dimshuffle',
                  concat_h=['input'], noise=0.1):

    '''
    Build fcn8 model as DAE
    '''

    net = {}
    pos = 0

    assert all([el in ['pool1', 'pool2', 'pool3', 'pool4', 'input']
                for el in concat_h])

    # Contracting path
    net['input'] = InputLayer((None, nb_in_channels, None, None),
                              input_mask_var)
    # Add noise
    # Noise
    if noise > 0:
        # net['noisy_input'] = GaussianNoiseLayerSoftmax(net['input'],
        #                                                sigma=noise)
        net['noisy_input'] = GaussianNoiseLayer(net['input'], sigma=noise)
        in_layer = 'noisy_input'
    else:
        in_layer = 'input'

    pos, out = model_helpers.concatenate(net, in_layer, concat_h,
                                         input_concat_h_vars, pos)

    # pool 1
    net['conv1_1'] = ConvLayer(
        net[out], 64, 3, pad=100, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad='same', flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)

    pos, out = model_helpers.concatenate(net, 'pool1', concat_h,
                                         input_concat_h_vars, pos)

    # pool 2
    net['conv2_1'] = ConvLayer(
        net[out], 128, 3, pad='same', flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad='same', flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)

    pos, out = model_helpers.concatenate(net, 'pool2', concat_h,
                                         input_concat_h_vars, pos)

    # pool 3
    net['conv3_1'] = ConvLayer(
        net[out], 256, 3, pad='same', flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad='same', flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad='same', flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)

    pos, out = model_helpers.concatenate(net, 'pool3', concat_h,
                                         input_concat_h_vars, pos)

    # pool 4
    net['conv4_1'] = ConvLayer(
        net[out], 512, 3, pad='same', flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad='same', flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad='same', flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)

    pos, out = model_helpers.concatenate(net, 'pool4', concat_h,
                                         input_concat_h_vars, pos)

    # pool 5
    net['conv5_1'] = ConvLayer(
        net[out], 512, 3, pad='same', flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad='same', flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad='same', flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)

    pos, out = model_helpers.concatenate(net, 'pool5', concat_h,
                                         input_concat_h_vars, pos)

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
                                   None), input_mask_var)

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
        pretrained = False
        with np.load(os.path.join(path_weights, model_name)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net['probs'], param_values)

    # In case we want to re-use the weights of an FCN8 model pretrained from images (not GT)
    if pretrained:
        if pascal:
            path_weights = '/data/lisatmp4/romerosa/itinf/models/camvid/pascal-fcn8s-tvg-dag.mat'
            if 'tvg' in path_weights:
                str_filter = 'f'
                str_bias = 'b'
            else:
                str_filter = '_filter'
                str_bias = '_bias'

            W = sio.loadmat(path_weights)

            # Load the parameter values into the net
            num_params = W.get('params').shape[1]
            str_ind = [''.join(x for x in concat if x.isdigit()) for concat in concat_h]
            list_of_lays = ['conv' + str(int(x)+1) + '_1' for x in str_ind if x]
            list_of_lays += ['conv1_1'] if nb_in_channels != 3 or 'input' in concat_h else []
            print list_of_lays

            for i in range(num_params):
                # Get layer name from the saved model
                name = str(W.get('params')[0][i][0])[3:-2]
                # Get parameter value
                param_value = W.get('params')[0][i][1]

                # Load weights
                if name.endswith(str_filter):
                    raw_name = name[:-len(str_filter)]

                    if raw_name not in list_of_lays:
                        print 'Copying weights for ' + raw_name
                        if 'score' not in raw_name and \
                           'upsample' not in raw_name and \
                           'final' not in raw_name and \
                           'probs' not in raw_name:

                            # print 'Initializing layer ' + raw_name
                            param_value = param_value.T
                            param_value = np.swapaxes(param_value, 2, 3)
                            net[raw_name].W.set_value(param_value)
                    else:
                        print 'Ignoring ' + raw_name

                # Load bias terms
                if name.endswith(str_bias):
                    raw_name = name[:-len(str_bias)]
                    if 'score' not in raw_name and \
                       'upsample' not in raw_name and \
                       'final' not in raw_name and \
                       'probs' not in raw_name:

                        param_value = np.squeeze(param_value)
                        net[raw_name].b.set_value(param_value)

        else:
            with np.load(os.path.join(pretrained_path, model_name)) as f:
                start = 0 if nb_in_channels == f['arr_%d' % 0].shape[1] \
                    else 2
                param_values = [f['arr_%d' % i] for i in range(start,
                                                               len(f.files))]
            all_layers = lasagne.layers.get_all_layers(net['probs'])
            all_layers = [l for l in all_layers if (not isinstance(l, InputLayer) and not isinstance(l, GaussianNoiseLayerSoftmax) and not isinstance(l,GaussianNoiseLayer))]
            all_layers = all_layers[1:] if start > 0 else all_layers
            # Freeze parameters after last concatenation layer
            last_concat = [idx for idx,l in enumerate(all_layers) if isinstance(l,ConcatLayer)][-1]
            count = 0
            for ixd, layer in enumerate(all_layers):
                layer_params = layer.get_params()
                for p in layer_params:
                    if hasattr(layer, 'input_layer') and not isinstance(layer.input_layer, ConcatLayer):
                        p.set_value(param_values[count])
                        if freeze:
                            model_helpers.freezeParameters(layer, single=True)
                    if isinstance(layer.input_layer, ConcatLayer) and idx == last_concat:
                        print('freezing')
                        freeze = True
                    count += 1

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(net['probs'])

    # Go back to 4D
    net['probs_reshape'] = ReshapeLayer(net['probs'], (laySize[0], laySize[1],
                                                       laySize[2], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'],
                                              (0, 3, 1, 2))

    return net[return_layer]
