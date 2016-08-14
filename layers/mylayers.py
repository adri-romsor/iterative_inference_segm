from lasagne.layers import MergeLayer
from lasagne.layers.merge import autocrop, autocrop_array_shapes


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
