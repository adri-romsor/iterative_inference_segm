import theano.tensor as T
import lasagne
from lasagne.layers import ReshapeLayer
from lasagne.layers import NonlinearityLayer, DimshuffleLayer, ElemwiseSumLayer
from layers.mylayers import CroppingLayer, DePool2D
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax, linear
import pdb

def UnpoolNet(incoming_net, net, p, unpool, n_classes, 
        incoming_layer, skip, unpool_old=True):
    # pdb.set_trace()
    if p == 1: 
        n_cl = n_classes
    else:
        laySize = incoming_net['pool'+str(p-1)].input_shape
        n_cl = laySize[1]
    print('Number of feature maps (out):', n_cl)

    if unpool_old:
        # Unpool
        # pdb.set_trace()
        net['up'+str(p)] = \
                DeconvLayer(incoming_net[incoming_layer] if p == unpool else
                        net['fused_up' + str(p+1)],
                        n_cl, 4, stride=2,
                        crop='valid', nonlinearity=linear)
        if skip and p > 1:
            # Merge
            net['fused_up'+str(p)] = \
                ElemwiseSumLayer((net['up'+str(p)],
                                  incoming_net['pool'+str(p-1)]),
                                cropping=[None, None, 'center', 'center'])
        else:
            # Crop
            net['fused_up'+str(p)] = \
                CroppingLayer((incoming_net['pool'+str(p-1) if p > 
                    1 else 'input'], 
                    net['up'+str(p)]),
                    merge_function=lambda input, deconv: 
                    deconv, cropping=[None, None, 'center', 'center'])
    else:
        # Depool
        net['up'+str(p)] = \
                DePool2D(incoming_net[incoming_layer] if p == unpool else
                         net['fused_up'+str(p+1)],
                         2, incoming_net['pool'+str(p)],
                         incoming_net['pool'+str(p)].input_layer)
        # Convolve
        net['up_conv'+str(p)] = \
                ConvLayer(net['up'+str(p)], n_cl, 3,
                          pad='valid', flip_filters=False)

        if skip and p > 1:
            # Merge
            net['fused_up'+str(p)] = \
                ElemwiseSumLayer((net['up_conv'+str(p)],
                                  incoming_net['pool'+str(p-1)]),
                                 cropping=[None, None, 'center', 'center'])

        else:
            # Crop
            net['fused_up'+str(p)] = \
                    CroppingLayer((incoming_net['pool'+str(p-1) if p > 
                        1 else 'input'], 
                        net['up_conv'+str(p)]),
                        merge_function=lambda input, deconv: 
                        deconv, cropping=[None, None, 'center', 'center'])

def buildFCN_up(incoming_net, incoming_layer, unpool,
                skip=False, n_classes=21):

    '''
    Build fcn decontracting path
    '''
    net = {}
    # Upsampling path

    # net['score'] = ConvLayer(incoming_net[incoming_layer], n_classes, 1,
    #                          pad='valid', flip_filters=True)

    p = 0

    for p in range(unpool, 0, -1):
        # Unpool and Crop if unpool_old=True or Depool and Conv
        UnpoolNet(incoming_net, net, p, unpool, n_classes, 
                incoming_layer, skip, unpool_old=True)

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
    # print('Input to last layer: ', net['probs_dimshuffle'].input_shape)
    print(net.keys())
    return net
