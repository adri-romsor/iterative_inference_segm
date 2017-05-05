import theano.tensor as T
import lasagne
from lasagne.layers import ReshapeLayer
from lasagne.layers import NonlinearityLayer, DimshuffleLayer, ElemwiseSumLayer, InverseLayer
from layers.mylayers import CroppingLayer, DePool2D
from lasagne.layers import Deconv2DLayer as DeconvLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax, linear


def UnpoolNet(incoming_net, net, p, unpool, n_classes,
              incoming_layer, skip, unpool_type='standard',
              layer_name=None):
    '''
    Add upsampling layer

    Parameters
    ----------
    incoming_net: contracting path network
    net: upsampling path network been built here
    p: int, the corresponding unpooling
    unpool: int, total number of unpoolings
    n_classes: int, number of classes
    incoming_layer: string, name of the last layer of the contracting path
    skip: bool, whether to skip connections
    unpool_type: string, unpooling type
    '''

    # last unpooling must have n_filters = number of classes
    if p == 1:
        n_cl = n_classes
    else:
        laySize = incoming_net['pool'+str(p-1)].input_shape
        n_cl = laySize[1]
    print('Number of feature maps (out):', n_cl)

    if unpool_type == 'standard':  # that would be standard deconv, with zeros
        # Unpool: the unpooling will use the last layer of the contracting path
        # if it is the first unpooling, otherwise it will use the last merged
        # layer (resulting from the previous unpooling)
        net['up'+str(p)] = \
                DeconvLayer(incoming_net[incoming_layer] if p == unpool else
                            net['fused_up' + str(p+1)],
                            n_cl, 4, stride=2,
                            crop='valid', nonlinearity=linear)
        # Add skip connection if required (sum)
        if skip and p > 1:
            # Merge
            net['fused_up'+str(p)] = \
                ElemwiseSumLayer((net['up'+str(p)],
                                  incoming_net['pool'+str(p-1)]),
                                 cropping=[None, None, 'center', 'center'],
                                 name=layer_name)
        else:
            # Crop to ensure the right output size
            net['fused_up'+str(p)] = \
                CroppingLayer((incoming_net['pool'+str(p-1) if p >
                                            1 else 'input'],
                               net['up'+str(p)]),
                              merge_function=lambda input, deconv:
                              deconv, cropping=[None, None,
                                                'center', 'center'],
                              name=layer_name)

    elif unpool_type == 'trackind' or unpool_type == 'inverse':
        # that would be index tracking as in SegNet
        # Depool: the unpooling will use the last layer of the contracting path
        # if it is the first unpooling, otherwise it will use the last merged
        # layer (resulting from the previous unpooling)
        if unpool_type == 'trackind':
            net['up'+str(p)] = \
                DePool2D(incoming_net[incoming_layer] if p == unpool else
                net['fused_up'+str(p+1)],
                2, incoming_net['pool'+str(p)],
                incoming_net['pool'+str(p)].input_layer)
        else:
            net['up'+str(p)] = \
                InverseLayer(incoming_net[incoming_layer] if p == unpool else
                    net['fused_up'+str(p+1)], incoming_net['pool'+str(p)])


        # Convolve
        net['up_conv'+str(p)] = \
            ConvLayer(net['up'+str(p)], n_cl, 3,
                      pad='same', flip_filters=False, nonlinearity=linear)
        # Add skip connection if required (sum)
        if skip and p > 1:
            # Merge
            net['fused_up'+str(p)] = \
                ElemwiseSumLayer((net['up_conv'+str(p)],
                                  incoming_net['pool'+str(p-1)]),
                                 cropping=[None, None, 'center', 'center'],
                                 name=layer_name)

        else:
            # Crop to ensure the right output size
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

    Parameters
    ----------
    incoming_net: contracting path network
    incoming_layer: string, name of last layer of the contracting path
    unpool: int, number of unpooling/upsampling layers we need
    skip: bool, whether to skip connections
    unpool_type: string, unpooling type
    n_classes: int, number of classes
    out_nonlin: output nonlinearity
    '''

    # Upsampling path
    net = {}

    # net['score'] = ConvLayer(incoming_net[incoming_layer], n_classes, 1,
    #                          pad='valid', flip_filters=True)

    # for loop to add upsampling layers
    for p in range(unpool, 0, -1):
        # Unpool and Crop if unpool_type='standard' or Depool and Conv
        if p == unpool-additional_pool+1 and ae_h:
            layer_name = 'h_hat'
        else:
            layer_name = None
        UnpoolNet(incoming_net, net, p, unpool, n_classes,
                  incoming_layer, skip, unpool_type=unpool_type,
                  layer_name=layer_name)

    # final dimshuffle, reshape and softmax
    net['final_dimshuffle'] = DimshuffleLayer(net['fused_up'+str(p) if
                                                  unpool > 0 else 'score'],
                                              (0, 2, 3, 1))
    laySize = lasagne.layers.get_output(net['final_dimshuffle']).shape
    net['final_reshape'] = ReshapeLayer(net['final_dimshuffle'],
                                        (T.prod(laySize[0:3]),
                                         laySize[3]))
    net['probs'] = NonlinearityLayer(net['final_reshape'],
                                     nonlinearity=out_nonlin)

    # go back to 4D
    net['probs_reshape'] = ReshapeLayer(net['probs'], (laySize[0], laySize[1],
                                                       laySize[2], n_classes))

    net['probs_dimshuffle'] = DimshuffleLayer(net['probs_reshape'],
                                              (0, 3, 1, 2))
    # print('Input to last layer: ', net['probs_dimshuffle'].input_shape)
    print(net.keys())
    return net
