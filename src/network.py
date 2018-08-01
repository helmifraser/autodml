import keras
import keras.backend as K
from keras import optimizers
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Activation, Dense, Dropout, Input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.training_utils import multi_gpu_model
from keras.layers.normalization import BatchNormalization

from coloured_print import printc
# import mobilenet_v2_noname

IMG_WIDTH = 224
IMG_HEIGHT = 134

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True) + 0.000001)

# params are [lr, d_1, d_2, d_3, dr]

def create_model(model_params=[0.001], seg = False, multi_gpu=False, gpus=2):
    img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='img')
    mobilenet = MobileNetV2(alpha=1.0,
                            depth_multiplier=1, include_top=False, pooling='max')(img)

    img_seg = []
    mobilenet_seg = []
    x = []
    if seg is True:
        img_seg = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='img_seg')
        mobilenet_seg = MobileNetV2(alpha=1.0,
                                depth_multiplier=1, include_top=False, pooling='max')(img_seg)
        x = keras.layers.concatenate([mobilenet_seg, mobilenet])

    # Default params
    model = []
    print(" ")
    print("Params:")
    print("Learning rate:", end=''); printc(" {}".format(model_params[0]), 'okgreen')
    print("Dense_1:", end=''); printc(" {}".format(model_params[1]), 'okgreen')
    print("Dense_2:", end=''); printc(" {}".format(model_params[2]), 'okgreen')
    print("Dense_3:", end=''); printc(" {}".format(model_params[3]), 'okgreen')
    print("Dropout rate:", end=''); printc(" {}".format(model_params[4]), 'okgreen')
    print(" ")
    if seg is True:
        x = Dropout(model_params[4])(x)
        x = Dense(model_params[1], use_bias=False)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
    else:
        x = Dropout(model_params[4])(mobilenet)
        x = Dense(model_params[1], input_shape=(1, 1280), use_bias=False)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

    # x = Dropout(model_params[4])(x)
    x = Dense(model_params[2], use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(model_params[4])(x)
    x = Dense(model_params[3], use_bias=False)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    # x = Dropout(model_params[4])(x)
    steer = Dense(1, use_bias=False)(x)
    steer = Activation('tanh')(steer)
    steer = BatchNormalization(name='steer')(steer)
    throttle = Dense(1, use_bias=False)(x)
    throttle = Activation('tanh')(throttle)
    throttle = BatchNormalization(name='throttle')(throttle)
    # brake = Dense(1, activation='tanh', name='brake')(x)
    if seg is True:
        model = Model(inputs=[img, img_seg], outputs=[steer, throttle])
    else:
        model = Model(inputs=img, outputs=[steer, throttle])

    for layers in model.layers:
        layers.trainable = True

    adam = optimizers.Adam(lr=model_params[0])

    print("Compiling model, will use ", end='')

    if multi_gpu is True:
        printc("{} GPUs".format(gpus), 'okgreen')
        parallel_model = multi_gpu_model(model, gpus=gpus)
        parallel_model.compile(optimizer=adam, loss=euclidean_distance_loss)
    else:
        printc("a single GPU")

    model.compile(optimizer=adam, loss=euclidean_distance_loss, loss_weights=[1, 0.2])

    model.summary()

    if multi_gpu is True:
        return parallel_model, model
    else:
        return model
