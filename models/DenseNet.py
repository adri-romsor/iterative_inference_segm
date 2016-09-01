import numpy as np
import theano.tensor as T

from lasagne.layers import (
    InputLayer, DropoutLayer, BatchNormLayer, ConcatLayer, NonlinearityLayer, Conv2DLayer, Pool2DLayer, Deconv2DLayer,
    Upscale2DLayer, ReshapeLayer, DimshuffleLayer, get_output, get_all_layers, get_output_shape, get_all_param_values)
from lasagne.nonlinearities import softmax, linear
from lasagne.init import HeUniform

import model_helpers


def buildDenseNet(nb_in_channels,
                  n_rows=None,
                  n_cols=None,
                  input_var=None,
                  n_classes=21,
                  n_filters_first_conv=12,
                  filter_size=3,
                  n_blocks=3,
                  growth_rate=12,
                  n_conv_per_block=3,
                  dropout_p=0.2,
                  pad_mode='same',
                  pool_mode='average',
                  dilated_convolution_index=None,
                  upsampling_mode='deconvolution',
                  n_filters_deconvolution='keep',
                  filter_size_deconvolution=4,
                  upsampling_block_mode=('dense', 12),
                  trainable=True):
    """
    We adapt DenseNet for segmentation
        Reference : https://arxiv.org/pdf/1608.06993.pdf
        GitHub : https://github.com/liuzhuang13/DenseNet

    The network is organized as follow :

    - Input
    - First convolution with n_filters_first_conv filters
    - Downsampling : n_block times Dense block + TransitionDown
    - Bottleneck : one Dense block
    - Upsampling : n_block times TransitionUp + Classic block or Dense block
    - 1x1 convolution and softmax

    In each Dense block, there n_conv_per_block convolution. Each convolution j is applied to an input stack of feature
    maps [x(0) ... x(j-1)]. It produces growth_rate new feature maps x(j) which is concatenated to the stack.
    Each convolution except the first is organized as follow : BatchNorm - ReLu - Conv - Dropout (if dropout_p > 0)

    TransitionDown first performs a 1x1 convolution which preserves the number of features maps in the stack and then
    average pooling

    TransitionUp : performs an upsampling and skip connections as in U-Net

    Classic block : during upsampling, we don't stack the features maps anymore, but perform usual convolution, It's
    more U-Net like

    Parameters
    ----------
    nb_in_channels : number of input channels
    input_var : tensor variable of the input
    n_classes : number of output channels
    n_rows and n_cols = optional, number of rows and columns of the input image
    n_filters_first_conv : number of filters to use for the first convolution of the network
    filter_size : filter size of the convolutions in the Dense block
    n_blocks : number of blocks in the downsampling path (= number of pooling we apply : if =5, to pool5)
    growth_rate : number of new feature maps in a Dense block during Downsampling
    n_conv_per_block_down : number of convolution in each block. Can be a list of size (2 * n_blocks + 1)
    dropout_p : dropout rate
    pad_mode : implemented : 'same'
    pool_mode : 'average' or 'max'
    dilated_convolution_index (int) : must be < n_blocks. If != None (default), apply dilated convolutions
    instead of pooling after this index. If = 3 and n_blocks = 5, there will be pool1, pool2 and pool3.
    upsampling_mode :
        if 'deconvolution' performs deconvolution to upsample. See deconvolution_mode
        if 'upscale' ... you shouldn't use that
        To implement : bilinear upsampling and WWAE
    n_filters_deconvolution :
        if 'keep', the deconvolution preserves the number of feature maps
        if int, the deconvolution outputs n_features maps. int should > n_classes
        if list, the deconvolution outputs list[i] features maps for i-th block. len(list) must be = n_blocks
    upsampling_block_mode : tuple(str, int or list)
        if str = 'dense' will use Dense block in the upsampling path
            if int : growth_rate
        if 'classic' will use Classic block in during the upsampling
            if int : number of filters for each convolution
            if list : n_filters for each block. len(list) must be equal to n_blocks. Ex : U-Net like : [256, 128, 64]
    trainable : freeze parameters if False
    """

    ##########################
    #    Check Parameters    #
    ##########################

    if dilated_convolution_index is None:
        dilated_convolution_index = n_blocks
    else:
        assert (0 < dilated_convolution_index < n_blocks)

    if isinstance(n_conv_per_block, list):
        assert (len(n_conv_per_block) == 2 * n_blocks + 1)
    else:
        n_conv_per_block = [n_conv_per_block] * (2 * n_blocks + 1)

    if isinstance(n_filters_deconvolution, list):
        assert len(n_filters_deconvolution) == n_blocks
    elif isinstance(n_filters_deconvolution, int):
        n_filters_deconvolution = [n_filters_deconvolution] * n_blocks

    assert isinstance(upsampling_block_mode, tuple)
    if upsampling_block_mode[0] == 'dense':
        growth_rate_up = upsampling_block_mode[1]
    elif upsampling_block_mode[0] == 'classic':
        if isinstance(upsampling_block_mode[1], list):
            assert len(upsampling_block_mode[1]) == n_blocks
            n_filters_up = upsampling_block_mode[1]
        elif isinstance(upsampling_block_mode[1], int):
            n_filters_up = [upsampling_block_mode[1]] * n_blocks

    #####################
    #    Layer utils    #
    #####################

    init_scheme = HeUniform()

    def BN_ReLu_Conv(input_stack, growth_rate, filter_size=filter_size):
        """
        Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout
        (if dropout_p > 0) on the inputs
         
        Returns
        -------
        
        growth_rate new features maps (the concatenation with inputs will be applied next)
        """

        l = BatchNormLayer(input_stack)  # TODO : test differents parameters for BN
        l = NonlinearityLayer(l)
        l = Conv2DLayer(l, growth_rate, filter_size, pad=pad_mode, W=init_scheme,
                        nonlinearity=linear, flip_filters=False)
        if dropout_p:
            l = DropoutLayer(l, dropout_p)
        return l

    def TransitionDown(input_stack, n_filters, block_index):
        """
        Apply succesivly BatchNormalization, ReLu nonlinearity, Convolution (filter size = 1),
        Dropout (if dropout_p > 0) and Pooling with a factor 2 
        If pool_mode = 'dilated' and block_index > apply_dilated_after_block_index,
        apply DilatedConvolution instead of pooling
        """

        l = BN_ReLu_Conv(input_stack, n_filters, filter_size=1)

        if block_index > dilated_convolution_index - 1:
            raise ValueError('Dilated convolutions not yet implemented')

        elif pool_mode == 'average':
            return Pool2DLayer(l, 2, mode='average_exc_pad')

        elif pool_mode == 'max':
            return Pool2DLayer(l, 2, mode='max')

    def TransitionUp(skip_connection, layer_to_upsample, n_filters_keep, block_index):
        """
        Performs upsampling on layer_to_upsample by a factor 2 and concatenate it with the skip_connection
        No upsampling if dilated convolution
        """

        if n_blocks - block_index - 1 > dilated_convolution_index - 1:
            # Dilated convolution
            return ConcatLayer([skip_connection, layer_to_upsample])

        if upsampling_mode == 'upscale':
            return ConcatLayer([skip_connection, Upscale2DLayer(layer_to_upsample, 2)],
                               cropping=[None, None, 'center', 'center'])

        elif upsampling_mode == 'deconvolution':
            if n_filters_deconvolution == 'keep':
                n_filters_out = n_filters_keep
            elif isinstance(n_filters_deconvolution, list):
                n_filters_out = n_filters_deconvolution[block_index]
            else:
                raise ValueError('Unknown n_filters_deconvolution value : ' + n_filters_deconvolution)

            l = Deconv2DLayer(layer_to_upsample, n_filters_out, filter_size_deconvolution, stride=2,
                              crop='valid', W=init_scheme, nonlinearity=linear)
            l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])
            return l

        elif upsampling_mode == 'bilinear':
            raise ValueError('Not yet implemented')
            # TODO ask david

        elif upsampling_mode == 'WWAE':
            raise ValueError('Not yet implemented')
            # TODO ask michal

        else:
            raise ValueError('Unknown upsampling_mode value : ' + upsampling_mode)

    #####################
    # Downsampling path #
    #####################

    inputs = InputLayer((None, nb_in_channels, n_rows, n_cols), input_var)

    # We perform a first convolution. All the features maps will be stored in the tensor stack
    stack = Conv2DLayer(inputs, n_filters_first_conv, filter_size, pad=pad_mode, W=init_scheme,
                        nonlinearity=linear, flip_filters=False)

    n_filters = n_filters_first_conv
    skip_connections = []
    for i in range(n_blocks):
        for j in range(n_conv_per_block[i]):
            l = BN_ReLu_Conv(stack, growth_rate)
            stack = ConcatLayer([stack, l])
            n_filters += growth_rate

        # As stack will be downsample, we don't upsample it again in the upsampling path but instead
        # store it as standard skip connections

        skip_connections.append(stack)
        stack = TransitionDown(stack, n_filters, i)

    skip_connections = skip_connections[::-1]

    #####################
    #     Bottleneck    #
    #####################

    # We store layers we'll have to upsample in a list
    layer_to_upsample = []
    for j in range(n_conv_per_block[n_blocks]):
        l = BN_ReLu_Conv(stack, growth_rate)
        layer_to_upsample.append(l)
        stack = ConcatLayer([stack, l])
        n_filters += growth_rate

    #######################
    #   Upsampling path   #
    #######################

    if upsampling_block_mode[0] == 'classic':
        l = ConcatLayer(layer_to_upsample)
        n_filters = growth_rate * n_conv_per_block[n_blocks]
        for i in range(n_blocks):
            n_filters_keep = n_filters_up[i] if i > 0 else n_filters
            l = TransitionUp(skip_connections[i], l, n_filters_keep, i)
            for j in range(n_conv_per_block[n_blocks + i + 1]):
                l = BN_ReLu_Conv(l, n_filters_up[i])

    elif upsampling_block_mode[0] == 'dense':
        for i in range(n_blocks):
            layer_to_upsample = ConcatLayer(layer_to_upsample)
            n_filters_keep = growth_rate * n_conv_per_block[n_blocks] if i == 0 \
                else growth_rate_up * n_conv_per_block[n_blocks + i]

            stack = TransitionUp(skip_connections[i], layer_to_upsample, n_filters_keep, i)
            layer_to_upsample = []
            for j in range(n_conv_per_block[n_blocks + i + 1]):
                l = BN_ReLu_Conv(stack, growth_rate_up)
                n_filters += growth_rate_up
                layer_to_upsample.append(l)
                stack = ConcatLayer([stack, l])
        l = stack

    else:
        raise ValueError('Unknown upsampling_block_mode value : ' + upsampling_block_mode[0])

    #####################
    #      Softmax      #
    #####################

    l = Conv2DLayer(l, n_classes, 1, nonlinearity=linear,
                    W=init_scheme, pad=pad_mode, flip_filters=False)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
    #     2. Apply softmax

    l = DimshuffleLayer(l, (0, 2, 3, 1))
    batch_size, n_rows, n_cols, _ = get_output(l).shape
    l = ReshapeLayer(l, (batch_size * n_rows * n_cols, n_classes))
    l = NonlinearityLayer(l, softmax)

    output_layer = l

    # Reshape the other way
    # l = ReshapeLayer(l, (batch_size, n_rows, n_cols, n_classes))
    # output_layer = DimshuffleLayer(l, (0, 3, 1, 2))

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(output_layer)

    return output_layer


def summary(cf):
    """
    Print a summary of the network associated to the config cf
    """
    # TODO : make this work with output_layer and image size as an input !

    output_layer = buildDenseNet(
        cf.nb_in_channels,
        cf.train_crop_size[0],
        cf.train_crop_size[1],
        None,
        cf.n_classes,
        cf.n_filters_first_conv,
        cf.filter_size,
        cf.n_blocks,
        cf.growth_rate,
        cf.n_conv_per_block,
        cf.dropout_p,
        cf.pad_mode,
        cf.pool_mode,
        cf.dilated_convolution_index,
        cf.upsampling_mode,
        cf.n_filters_deconvolution,
        cf.filter_size_deconvolution,
        cf.upsampling_block_mode,
        cf.trainable)

    layer_list = get_all_layers(output_layer)

    for layer_type in [DropoutLayer, NonlinearityLayer, BatchNormLayer, ReshapeLayer, DimshuffleLayer]:
        layer_list = filter(lambda x: not isinstance(x, layer_type), layer_list)
    output_shape_list = map(get_output_shape, layer_list)

    layer_name = lambda s: str(s).split('.')[3].split(' ')[0]

    print('-' * 75)
    print '\n   Layer       Output shape           W shape \n'

    for i, (layer, output_shape) in enumerate(zip(layer_list, output_shape_list)):
        if isinstance(layer, Conv2DLayer):
            input_shape = layer.W.get_value().shape
        else:
            input_shape = ''
        print '{}. {}  {}  {}'.format(i, layer_name(layer), output_shape, input_shape)

    print '\nNumber of conv layers : {}'.format(len(filter(lambda x: isinstance(x, Conv2DLayer), layer_list)))
    print 'Number of parameters : {}'.format(np.sum(map(np.size, get_all_param_values(output_layer))))
    print('-' * 75)


if __name__ == '__main__':
    from theano import function
    from lasagne.layers import get_output

    x = T.tensor4('x', dtype='float32')

    output_layer = buildDenseNet(nb_in_channels=3,
                                 input_var=x,
                                 n_classes=21,
                                 n_filters_first_conv=12,
                                 filter_size=3,
                                 n_blocks=3,
                                 growth_rate=12,
                                 n_conv_per_block=3,
                                 dropout_p=0.2,
                                 pad_mode='same',
                                 pool_mode='average',
                                 dilated_convolution_index=None,
                                 upsampling_mode='deconvolution',
                                 n_filters_deconvolution='keep',
                                 upsampling_block_mode='classic',
                                 trainable=True)

    output_tensor = get_output(output_layer)

    # f = function([x], output_tensor)
    #
    # x_arr = np.ones((10, 3, 100, 100), dtype='float32')
    # y_arr = f(x_arr)
    # print y_arr
    # print y_arr.shape
