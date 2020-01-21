from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import os



def distillation_cifar10_model():
    input_shape=(32, 32, 3)
    num_classes = 10
    n = 3
    depth = n * 6 + 2
    return resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)

# def resnet32_cifar10_model():
#     input_shape=(32, 32, 3)
#     num_classes = 10
#     n = 5
#     depth = n * 6 + 2
#     return resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
#
# def resnet44_cifar10_model():
#     input_shape=(32, 32, 3)
#     num_classes = 10
#     n = 7
#     depth = n * 6 + 2
#     return resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
#
# def resnet56_cifar10_model():
#     input_shape=(32, 32, 3)
#     num_classes = 10
#     n = 9
#     depth = n * 6 + 2
#     return resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)
#
#
# def resnet110_cifar10_model():
#     input_shape=(32, 32, 3)
#     num_classes = 10
#     n = 18
#     depth = n * 6 + 2
#     return resnet_v1(input_shape=input_shape, depth=depth, num_classes=num_classes)


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):


    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

        x = conv(x)

    return x





def resnet_v1(input_shape, depth, num_classes):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

        # Start model definition.

    num_filters = 16


    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)

    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units

    for stack in range(3):

        for res_block in range(num_res_blocks):

            strides = 1

            if stack > 0 and res_block == 0:  # first layer but not first stack

                strides = 2  # downsample

            y = resnet_layer(inputs=x,

                             num_filters=num_filters,

                             strides=strides)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters,

                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack

                # linear projection residual shortcut connection to match

                # changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = keras.layers.add([x, y])

            x = Activation('relu')(x)

        num_filters *= 2

    # Add classifier on top.

    # v1 does not use BN after last shortcut connection-ReLU

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,

                    activation='softmax',

                    kernel_initializer='he_normal')(y)

    # Instantiate model.

    model = Model(inputs=inputs, outputs=outputs)

    return model







