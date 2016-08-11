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
