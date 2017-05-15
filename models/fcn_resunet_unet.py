import keras.backend as K
from keras.models import Model
from keras.layers import (Input, merge,
                          Convolution2D,
                          MaxPooling2D,
                          UpSampling2D,
                          Permute,
                          Activation)
from keras.regularizers import l2

def _softmax(x):
    """
    Softmax that works on ND inputs.
    """
    e = K.exp(x - K.max(x, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s


def build_unet(img_shape,
               regularize_weights=False,
               nclasses=2,
               nb_feat = 64,
               preprocessing=False):

    if regularize_weights:
        print("regularizing the weights")
        l2_reg = regularize_weights
    else:
        l2_reg = 0.

    inputs = Input(img_shape)
    conv1 = Convolution2D(nb_feat, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(inputs)
    conv1 = Convolution2D(nb_feat, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(nb_feat*2, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(pool1)
    conv2 = Convolution2D(nb_feat*2, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(nb_feat*4, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(pool2)
    conv3 = Convolution2D(nb_feat*4, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(nb_feat*8, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(pool3)
    conv4 = Convolution2D(nb_feat*8, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(nb_feat*16, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(pool4)
    conv5 = Convolution2D(nb_feat*16, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat',
                concat_axis=1)
    up6 = Convolution2D(nb_feat*8, 2, 2, activation='linear', border_mode='same',
                          W_regularizer=l2(l2_reg))(up6)
    conv6 = Convolution2D(nb_feat*8, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(up6)
    conv6 = Convolution2D(nb_feat*8, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat',
                concat_axis=1)
    up7 = Convolution2D(nb_feat*4, 2, 2, activation='linear', border_mode='same',
                          W_regularizer=l2(l2_reg))(up7)
    conv7 = Convolution2D(nb_feat*4, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(up7)
    conv7 = Convolution2D(nb_feat*4, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat',
                concat_axis=1)
    up8 = Convolution2D(nb_feat*2, 2, 2, activation='linear', border_mode='same',
                          W_regularizer=l2(l2_reg))(up8)
    conv8 = Convolution2D(nb_feat*2, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(up8)
    conv8 = Convolution2D(nb_feat*2, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat',
                concat_axis=1)
    up9 = Convolution2D(nb_feat, 2, 2, activation='linear', border_mode='same',
                          W_regularizer=l2(l2_reg))(up9)
    conv9 = Convolution2D(nb_feat, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(up9)
    conv9 = Convolution2D(nb_feat, 3, 3, activation='relu', border_mode='same',
                          W_regularizer=l2(l2_reg))(conv9)

    if preprocessing:
        conv10 = Convolution2D(nclasses, 1, 1, border_mode='same',
                               W_regularizer=l2(l2_reg))(conv9)
        model = Model(input=inputs, output=conv10)
        return model

    else:

        conv10 = Convolution2D(nclasses, 1, 1, border_mode='same',
                               W_regularizer=l2(l2_reg))(conv9)

        pre_sm = Permute((2,3,1))(conv10)

        if nclasses==1:
            output = Activation('sigmoid')(pre_sm)
        else:
            output = Activation(_softmax)(pre_sm)
        output = Permute((3,1,2))(output)

        model = Model(input=inputs, output=output)

        return model
