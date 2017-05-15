import keras.backend as K
from keras.models import Model
from keras.layers.convolutional import (Convolution2D)
from keras.layers.core import Dropout
from keras.layers import Input
from keras.regularizers import l2

from unet import build_unet

def build_preprocessing(img_shape,
                        nb_filter=64,
                        kernel_size=3,
                        nb_layers=3,
                        regularize_weights=False,
                        pre_unet=False,
                        output_nb_filter=1):

    if pre_unet:
        model = build_unet(img_shape=img_shape,
                           regularize_weights=regularize_weights,
                           nb_feat=nb_filter,
                           nclasses=output_nb_filter,
                           preprocessing=True)
        return model

    else:

        if regularize_weights:
            print("regularizing the weights")
            l2_reg = regularize_weights
        else:
            l2_reg = 0.


        input = Input(img_shape)

        for i in range(nb_layers):
            if i==0:
                conv_tmp = Convolution2D(nb_filter, kernel_size, kernel_size,
                                         activation='relu',
                                         border_mode='same',
                                         dim_ordering='th',
                                         name=('conv_pre_' + str(i)),
                                         W_regularizer=l2(l2_reg))(input)
            else:
                conv_tmp = Convolution2D(nb_filter, kernel_size, kernel_size,
                                         activation='relu',
                                         border_mode='same',
                                         dim_ordering='th',
                                         name=('conv_pre_' + str(i)),
                                         W_regularizer=l2(l2_reg))(conv_tmp)

        model = Model(input=input, output=conv_tmp)
        return model
