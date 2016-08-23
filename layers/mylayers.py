from lasagne.layers import MergeLayer
from lasagne.layers.merge import autocrop, autocrop_array_shapes
from lasagne.layers.pool import Upscale2DLayer
import theano.tensor as T
from lasagne.utils import as_tuple


class ElemwiseMergeLayer(MergeLayer):
    """
    This layer performs an elementwise merge of its input layers.
    It requires all input layers to have the same output shape.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal
    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.
    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`
    See Also
    --------
    ElemwiseSumLayer : Shortcut for sum layer.
    """

    def __init__(self, incomings, merge_function, cropping=None, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = autocrop_array_shapes(input_shapes, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = input_shapes[1]

        return output_shape

    def get_output_for(self, inputs, **kwargs):
        inputs = autocrop(inputs, self.cropping)
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output

class DePool2D(Upscale2DLayer):
   ''' Code taken from:
   https://github.com/nanopony/keras-convautoencoder/blob/master/autoencoder_layers.py

    Simplar to Upscale2DLayer, yet traverse only maxpooled elements
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.
    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
    '''

    def __init__(self, incoming, scale_factor, **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)
        self.scale_factor = as_tuple(scale_factor, 2)
        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))    

    def get_output_for(self, inputs, train=False, **kwargs):
        a, b = self.scale_factor
        upscaled = inputs.output(train)
        if b > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, 3)
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, 2)
        f = T.grad(T.sum(inputs.output(train)),
                wrt=inputs.input(train)) * upscaled
        
        return f 
