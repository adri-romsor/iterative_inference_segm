import theano
import lasagne


def freezeParameters(net, single=True):
    all_layers = lasagne.layers.get_all_layers(net)

    if single:
        all_layers = [all_layers[-1]]

    for layer in all_layers:
        layer_params = layer.get_params()
        for p in layer_params:
            try:
                layer.params[p].remove('trainable')
            except KeyError:
                pass


def unfreezeParameters(net, single=True):
    all_layers = lasagne.layers.get_all_layers(net)

    if single:
        all_layers = [all_layers[-1]]

    for layer in all_layers:
        layer_params = layer.get_params()
        for p in layer_params:
            try:
                layer.params[p].add('trainable')
            except KeyError:
                pass

def softmax4D(x):
    """
    Softmax activation function for a 4D tensor of shape (b, c, 0, 1)
    """
    # Compute softmax activation
    stable_x = x - theano.gradient.zero_grad(x.max(1, keepdims=True))
    exp_x = stable_x.exp()
    softmax_x = exp_x / exp_x.sum(1)[:, None, :, :]

    return softmax_x
