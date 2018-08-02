from keras.layers import Convolution2D, Input
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD


import tensorflow as tf

from keras import backend as K
K.set_image_dim_ordering('tf')

IMG_WIDTH = 224
IMG_HEIGHT = 224


def atan_layer(x):
    print(x, tf.mul(tf.atan(x), 2))
    return tf.mul(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def atan(x):
    return tf.atan(x)


def create_model(params):

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    conv_1 = Convolution2D(24, 5, 5, activation='relu',
                           name='conv_1', subsample=(2, 2))(inputs)

    conv_2 = Convolution2D(36, 5, 5, activation='relu',
                           name='conv_2', subsample=(2, 2))(conv_1)

    conv_3 = Convolution2D(48, 5, 5, activation='relu',
                           name='conv_3', subsample=(2, 2))(conv_2)
    conv_3 = Dropout(params[-1])(conv_3)

    conv_4 = Convolution2D(64, 3, 3, activation='relu',
                           name='conv_4', subsample=(1, 1))(conv_3)

    conv_5 = Convolution2D(64, 3, 3, activation='relu',
                           name='conv_5', subsample=(1, 1))(conv_4)

    flat = Flatten()(conv_5)

    dense_1 = Dense(1164)(flat)
    dense_1 = Dropout(params[-1])(flat)

    dense_2 = Dense(params[1], activation='relu')(dense_1)
    dense_2 = Dropout(params[-1])(dense_2)

    dense_3 = Dense(params[2], activation='relu')(dense_2)
    dense_3 = Dropout(params[-1])(dense_3)
    #
    # dense_4 = Dense(params[3], activation='relu')(dense_3)
    # dense_4 = Dropout(params[-1])(dense_4)
    #
    # dense_5 = Dense(params[4], activation='relu')(dense_4)
    # dense_5 = Dropout(params[-1])(dense_5)


    # final = Dense(1, activation=atan)(dense_4)
    steer = Dense(1, activation=atan, name='steer')(dense_3)
    throttle = Dense(1, activation=atan, name='throttle')(dense_3)

    #angle = Lambda(lambda x: tf.mul(tf.atan(x), 2))(final)

    model = Model(input=inputs, output=[steer, throttle])
    model.compile(
        optimizer=SGD(lr=params[0], momentum=.9),
        loss='mse')
    model.summary()

    return model
