import numpy as np
import theano.tensor as T

from lasagne.layers import (
    InputLayer, DropoutLayer, BatchNormLayer, ConcatLayer, NonlinearityLayer, Conv2DLayer, Pool2DLayer, Deconv2DLayer,
    Upscale2DLayer, ReshapeLayer, DimshuffleLayer, get_output)
from lasagne.nonlinearities import softmax, linear
from lasagne.init import HeUniform

import model_helpers


def buildDenseNet(nb_in_channels,
                  input_var=None,
                  n_classes=21,
                  n_filters_first_conv=12,
                  filter_size=3,
                  n_blocks=3,
                  growth_rate_down=12,
                  growth_rate_up=12,
                  n_conv_per_block_down=3,
                  n_conv_per_block_up=3,
                  dropout_p=0.2,
                  pad_mode='same',
                  pool_mode='average',
                  upsampling_mode='upscale',
                  trainable=True):
    """
    We adapt DenseNet for segmentation

    Reference : https://arxiv.org/pdf/1608.06993.pdf
    GitHub : https://github.com/liuzhuang13/DenseNet

    Parameters
    ----------
    nb_in_channels : number of input channels
    input_var : tensor variable of the input
    n_classes : number of classes
    n_filters_first_conv : number of filters to use for the first convolution of the network
    filter_size : 3... as usual...
    n_blocks :
    growth_rate_down
    growth_rate_up
    n_conv_per_block_down
    n_conv_per_block_up
    dropout_p
    pad_mode
    pool_mode
    upsampling_mode
    trainable

    Returns
    -------

    """

    # TODO : check pad options

    #####################
    #    Layer utils    #
    #####################

    init_scheme = HeUniform()

    def BN_ReLu_Conv(inputs, growth_rate, filter_size=filter_size):
        """
        Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout
        (if dropout_p > 0) on the inputs, to produce growth_rate new feature maps
        """

        l = BatchNormLayer(inputs)
        # TODO : check parameters of BN
        l = NonlinearityLayer(l)
        l = Conv2DLayer(l, growth_rate, filter_size, pad=pad_mode, W=init_scheme, nonlinearity=linear)
        if dropout_p:
            l = DropoutLayer(l, dropout_p)
        return l

    def TransitionLayer(inputs):
        """
        Apply succesivly BatchNormalization, ReLu nonlinearity, Convolution (filter size = 1),
        Dropout (if dropout_p > 0) and Pooling with a factor 2, except if pool_mode = 'atrous'
        """

        # TODO : what is the justification of 1x1 convolution !?
        l = BN_ReLu_Conv(inputs, growth_rate_down, filter_size=1)

        if pool_mode == 'average':
            return Pool2DLayer(l, 2, mode='average_exc_pad')

        elif pool_mode == 'max':
            return Pool2DLayer(l, 2, mode='max')

        elif pool_mode == 'atrous':
            raise ValueError('Not yet implemented')

        else:
            raise ValueError('Unknown pool_mode value : ' + pool_mode)

    def UpsamplingLayer(inputs, n_filters=None, filter_size=4):
        """
        Performs upsampling on the input by a factor 2
        If pool_mode = 'atrous' returns inputs

        Parameters
        ----------
        inputs
        n_filters : if upsampling_mode = 'deconvolution', must be equal to the number of features maps to deconvolve
        filter_size : filter_size for the deconvolution
        """

        # TODO dimensions impaires

        if pool_mode == 'atrous':
            return inputs

        if upsampling_mode == 'upscale':
            return Upscale2DLayer(inputs, 2)
            # TODO : ajout convolution

        elif upsampling_mode == 'deconvolution':
            l = Deconv2DLayer(inputs, n_filters, filter_size, stride=2,
                                 crop='valid', W=init_scheme, nonlinearity=linear)


        elif upsampling_mode == 'bilinear':
            raise ValueError('Not yet implemented')

        elif upsampling_mode == 'WWAE':
            raise ValueError('Not yet implemented')

        else:
            raise ValueError('Unknown upsampling_mode value : ' + upsampling_mode)

    #####################
    # Downsampling path #
    #####################

    inputs = InputLayer((None, nb_in_channels, None, None), input_var)

    # We perform a first convolution. All the features maps wil be store in the tensor concatenation
    concatenation = Conv2DLayer(inputs, n_filters_first_conv, filter_size, pad=pad_mode, W=init_scheme,
                                nonlinearity=linear)

    n_filters = n_filters_first_conv
    skip_connections = []
    for i in range(n_blocks):
        for j in range(n_conv_per_block_down):
            l = BN_ReLu_Conv(concatenation, growth_rate_down)
            concatenation = ConcatLayer([concatenation, l])
            n_filters += growth_rate_down

        # As concatenation will be downsample, we don't upsample it again in the upsampling path but instead
        # store it as standard skip connections

        skip_connections.append(concatenation)
        concatenation = TransitionLayer(concatenation)
        n_filters += growth_rate_down

    skip_connections = skip_connections[::-1]

    #####################
    #     Bottleneck    #
    #####################

    # We store layers we'll have to upsample in a list
    layers_to_upsample = []
    for j in range(n_conv_per_block_down):
        l = BN_ReLu_Conv(concatenation, growth_rate_down)
        layers_to_upsample.append(l)
        concatenation = ConcatLayer([concatenation, l])
        n_filters += growth_rate_down

    #####################
    #  UpsamplingLayer path  #
    #####################

    for i in range(n_blocks):
        layer_to_concat = UpsamplingLayer(ConcatLayer(layers_to_upsample), n_filters)
        concatenation = ConcatLayer([skip_connections[i], layer_to_concat])
        layers_to_upsample = []
        for j in range(n_conv_per_block_up):
            l = BN_ReLu_Conv(concatenation, growth_rate_up)
            n_filters += growth_rate_up
            layers_to_upsample.append(l)
            concatenation = ConcatLayer([concatenation, l])

    #####################
    #      Softmax      #
    #####################

    l = Conv2DLayer(concatenation, n_classes, 1, nonlinearity=linear, W=init_scheme, pad=pad_mode)

    # We perform the softmax nonlinearity in 2 steps :
    #     1. Reshape from (batch_size, n_classes, n_rows, n_cols) to (batch_size  * n_rows * n_cols, n_classes)
    #     2. Apply softmax
    # TODO : why don't we reshape again ?

    l = DimshuffleLayer(l, (0, 2, 3, 1))
    batch_size, n_rows, n_cols, _ = get_output(l).shape
    l = ReshapeLayer(l, (batch_size * n_rows * n_cols, n_classes))
    l = NonlinearityLayer(l, softmax)

    l = ReshapeLayer(l, (batch_size, n_rows, n_cols, n_classes))
    output_layer = DimshuffleLayer(l, (0, 3, 1, 2))
    # output_layer = l

    # Do not train
    if not trainable:
        model_helpers.freezeParameters(output_layer)

    print('Number of features maps before softmax = {}'.format(n_filters))
    print('Number of conv layers = {}'.format(1 + (n_conv_per_block_down + n_conv_per_block_up + 1) * n_blocks))

    return output_layer


if __name__ == '__main__':
    from theano import function
    from lasagne.layers import get_output

    x = T.tensor4('x', dtype='float32')

    output_layer = buildDenseNet(nb_in_channels=3,
                                 input_var=x,
                                 n_classes=21,
                                 n_filters_first_conv=12,
                                 filter_size=3,
                                 n_blocks=2,
                                 growth_rate_down=12,
                                 growth_rate_up=12,
                                 n_conv_per_block_down=3,
                                 n_conv_per_block_up=3,
                                 dropout_p=0.2,
                                 pad_mode='same',
                                 pool_mode='average',
                                 upsampling_mode='upscale',
                                 trainable=True)

    output_tensor = get_output(output_layer)
    f = function([x], output_tensor)

    x_arr = np.ones((10, 3, 32, 32), dtype='float32')
    y_arr = f(x_arr)
    print y_arr
    print y_arr.shape
