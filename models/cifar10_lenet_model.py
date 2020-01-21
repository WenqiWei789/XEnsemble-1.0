from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import MaxPooling2D, Conv2D, BatchNormalization


# def carlini_mnist_model(logits=True, input_range_type=2, pre_filter=lambda x:x):
#     input_shape=(28, 28, 1)
#     nb_filters = 32
#     nb_denses = [200,200,10]##original 200 200 10
#     return carlini_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter)


def lenet_cifar10_model(logits=True, input_range_type=2, pre_filter=lambda x:x):
    input_shape=(32, 32, 3)
    nb_filters = 64
    nb_denses = [256,256,10]
    return lenet_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter)


def lenet_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter):
    """
    :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
    :params input_range_type: the expected input range, {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
    """

    model = Sequential()

    if input_range_type == 1:
        # The input data range is [0, 1]. 
        # Convert to [-0.5,0.5] by x-0.5.
        scaler = lambda x: x-0.5
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5].
        # Don't need to do scaling for carlini models, as it is the input range by default.
        scaler = lambda x: x
    elif input_range_type == 3:
        # The input data range is [-1, 1]. Convert to [-0.5,0.5] by x/2.
        scaler = lambda x: x/2

    drop_out = 0.4
    model.add(Lambda(scaler, input_shape=input_shape))
    model.add(Lambda(pre_filter, output_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_out))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_out))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(10))



    if not logits:
        model.add(Activation('softmax'))

    return model