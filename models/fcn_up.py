import theano.tensor as T
import lasagne
from lasagne.layers import ReshapeLayer
from lasagne.layers import NonlinearityLayer, DimshuffleLayer, ElemwiseSumLayer
from layers.mylayers import CroppingLayer, DePool2D
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax, linear


def UnpoolNet(incoming_net, net, p, unpool, n_classes,
              incoming_layer, skip, unpool_type='standard',
              layer_name=None):
    # pdb.set_trace()
    if p == 1:
        n_cl = n_classes
    else:
        laySize = incoming_net['pool'+str(p-1)].input_shape
        n_cl = laySize[1]
    print('Number of feature maps (out):', n_cl)

    if unpool_type == 'standard':
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
                                 cropping=[None, None, 'center', 'center'],
                                 name=layer_name)
        else:
            # Crop
            net['fused_up'+str(p)] = \
                CroppingLayer((incoming_net['pool'+str(p-1) if p >
                                            1 else 'input'],
                               net['up'+str(p)]),
                              merge_function=lambda input, deconv:
                              deconv, cropping=[None, None,
                                                'center', 'center'],
                              name=layer_name)

    elif unpool_type == 'trackind':
        # Depool
        net['up'+str(p)] = \
                DePool2D(incoming_net[incoming_layer] if p == unpool else
                         net['fused_up'+str(p+1)],
                         2, incoming_net['pool'+str(p)],
                         incoming_net['pool'+str(p)].input_layer)
        # Convolve
        net['up_conv'+str(p)] = \
            ConvLayer(net['up'+str(p)], n_cl, 3,
                      pad='same', flip_filters=False, nonlinearity=linear)

        if skip and p > 1:
            # Merge
            net['fused_up'+str(p)] = \
                ElemwiseSumLayer((net['up_conv'+str(p)],
                                  incoming_net['pool'+str(p-1)]),
                                 cropping=[None, None, 'center', 'center'],
                                 name=layer_name)

        else:
            # Crop
            net['fused_up'+str(p)] = \
                    CroppingLayer((incoming_net['pool'+str(p-1) if p >
                                                1 else 'input'],
                                   net['up_conv'+str(p)]),
                                  merge_function=lambda input, deconv:
                                  deconv, cropping=[None, None,
                                                    'center', 'center'],
                                  name=layer_name)
    else:
        raise ValueError('Unkown unpool type')


def buildFCN_up(incoming_net, incoming_layer, unpool,
                skip=False, unpool_type='standard',
                n_classes=21, out_nonlin=linear,
                additional_pool=0, ae_h=False):

    '''
    Build fcn decontracting path
    '''
    net = {}
    # Upsampling path
    # net['score'] = ConvLayer(incoming_net[incoming_layer], n_classes, 1,
    #                          pad='valid', flip_filters=True)
    for p in range(unpool, 0, -1):
        # Unpool and Crop if unpool_type='standard' or Depool and Conv
        if p == unpool-additional_pool+1 and ae_h:
            layer_name = 'h_hat'
        else:
            layer_name = None

        UnpoolNet(incoming_net, net, p, unpool, n_classes,
                  incoming_layer, skip, unpool_type=unpool_type,
                  layer_name=layer_name)

    # Final dimshuffle, reshape and softmax
    net['final_dimshuffle'] = DimshuffleLayer(net['fused_up'+str(p) if
                                                  unpool > 0 else 'score'],
                                              (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = ReshapeLayer(net['final_dimshuffle'],
                                        (T.prod(laySize[0:3]),
                                         laySize[3]))
    net['probs'] = NonlinearityLayer(net['final_reshape'],
                                     nonlinearity=out_nonlin)

    # Go back to 4D
    net['probs_reshape'] = ReshapeLayer(net['probs'], (laySize[0], laySize[1],
                                                       laySize[2], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'],
                                              (0, 3, 1, 2))
    # print('Input to last layer: ', net['probs_dimshuffle'].input_shape)
    print(net.keys())
    return net
