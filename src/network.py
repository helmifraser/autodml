import keras
import keras.backend as K
from keras import optimizers
from keras import losses
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Model
from keras.utils.training_utils import multi_gpu_model
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model

from coloured_print import printc

IMG_WIDTH = 224
IMG_HEIGHT = 224


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True) + 0.000001)


def create_branch(x, params, name=None):

    y = dense_block(x, params[0])

    y = dense_block(y, params[1])

    if name is not None:
        y = dense_block(y, 1, name=name, activation='tanh')
    else:
        y = dense_block(y, 1, activation='tanh')

    return y


def dense_block(x, neurons, name=None, activation='relu', bias=False):
    y = Dense(neurons, use_bias=bias)(x)
    y = Activation(activation)(y)

    if name is not None:
        y = BatchNormalization(name=name)(y)
    else:
        y = BatchNormalization()(y)

    return y


def print_params(model_params):
    print(" ")
    print("Params:")
    print("Learning rate:", end='')
    printc(" {}".format(model_params[0]), 'okgreen')

    for id, value in enumerate(model_params[1:-1]):
        print("Dense_{}:".format(id), end='')
        printc(" {}".format(value), 'okgreen')

    print("Dropout rate:", end='')
    printc(" {}".format(model_params[-1]), 'okgreen')
    print(" ")

# params are [lr, d_1, d_2, d_3, dr]

def create_model(model_params, seg=False, multi_gpu=False, gpus=2):
    img_seg = []
    mobilenet_seg = []
    x = []
    model = []

    img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='img')

    mobilenet = MobileNetV2(alpha=1.0,
                            depth_multiplier=1,
                            include_top=False,
                            weights='imagenet',
                            pooling='max')(img)

    print_params(model_params)

    if seg is True:
        img_seg = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='img_seg')
        mobilenet_seg = MobileNetV2(alpha=1.0,
                                    depth_multiplier=1,
                                    include_top=False,
                                    pooling='max')(img_seg)
        x = keras.layers.concatenate([mobilenet_seg, mobilenet])

        steer = create_branch(x, model_params[1:-1], name='steer')
        throttle = create_branch(x, model_params[1:-1], name='throttle')
        model = Model(inputs=[img, img_seg], outputs=[steer, throttle])
    else:
        steer = create_branch(mobilenet, model_params[1:-1], name='steer')
        throttle = create_branch(mobilenet, model_params[1:-1], name='throttle')
        model = Model(inputs=img, outputs=[steer, throttle])

    for layers in model.layers[:2]:
        layers.trainable = False

    adam = optimizers.Adam(lr=model_params[0])

    print("Compiling model, will use ", end='')

    if multi_gpu is True:
        printc("{} GPUs".format(gpus), 'okgreen')
        parallel_model = multi_gpu_model(model, gpus=gpus)
        parallel_model.compile(optimizer=adam, loss=losses.logcosh)
    else:
        printc("a single GPU")

    model.compile(optimizer=adam, loss=losses.logcosh,
                  loss_weights=[1, 1])

    model.summary()

    if multi_gpu is True:
        return parallel_model, model
    else:
        return model
