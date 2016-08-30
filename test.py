# def BN_ReLu_Conv(net, inputs, growth_rate, idx_block, idx_conv):
#
#     id = dict((layer_name, '{}_{}_{}'.format(layer_name, idx_block, idx_conv))
#                  for layer_name in ['bn', 'relu', 'conv', 'drop'])
#
#     net[id['bn']] = BatchNormLayer(inputs)
#     net[id['relu']] = NonlinearityLayer(net[id['bn']])
#     net[id['conv']] = Conv2DLayer(net[id['relu']], growth_rate, filter_size, pad=pad,
#                                   W=init_scheme, nonlinearity=linear)
#     if dropout_p:
#         net[id['drop']] = DropoutLayer(net[id['conv']], dropout_p)