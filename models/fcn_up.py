import theano.tensor as T
import lasagne
from lasagne.layers import ReshapeLayer
from lasagne.layers import NonlinearityLayer, DimshuffleLayer, ElemwiseSumLayer
from layers.mylayers import ElemwiseMergeLayer
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax, linear


def buildFCN_up(incoming_net, incoming_layer, unpool,
                skip=False, n_classes=21):

    '''
    Build fcn decontracting path
    '''

    net = {}

    # Upsampling path

    net['score'] = ConvLayer(incoming_net[incoming_layer], n_classes, 1,
                             pad='valid', flip_filters=False)

    p = 0

    for p in range(unpool, 0, -1):
        # Unpool
        net['up'+str(p)] = \
            DeconvLayer(net['score'] if p == unpool else
                        net['fused_up' + str(p+1)],
                        n_classes, 4, stride=2,
                        crop='valid', nonlinearity=linear)
        if skip and p > 1:
            # Conv to reduce dimensionality of incoming layer
            net['score_pool'+str(p-1)] = \
                ConvLayer(incoming_net['pool'+str(p-1)],
                          n_classes, 1, pad='same')

            # Merge and crop
            net['fused_up'+str(p)] = \
                ElemwiseSumLayer((net['up'+str(p)],
                                  net['score_pool'+str(p-1)]),
                                 cropping=[None, None, 'center', 'center'])
        else:
            # Crop
            net['fused_up'+str(p)] = \
                ElemwiseMergeLayer((incoming_net['pool'+str(p-1) if p >
                                                 1 else 'input'],
                                    net['up'+str(p)]),
                                   merge_function=lambda input, deconv:
                                   deconv, cropping=[None, None,
                                                     'center', 'center'])

    # Final dimshuffle, reshape and softmax
    net['final_dimshuffle'] = DimshuffleLayer(net['fused_up'+str(p) if
                                                  unpool > 0 else 'score'],
                                              (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = ReshapeLayer(net['final_dimshuffle'],
                                        (T.prod(laySize[0:3]),
                                         laySize[3]))
    net['probs'] = NonlinearityLayer(net['final_reshape'],
                                     nonlinearity=softmax)

    # Go back to 4D
    net['probs_reshape'] = ReshapeLayer(net['probs'], (laySize[0], laySize[1],
                                                       laySize[2], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'],
                                              (0, 3, 1, 2))

    return net
