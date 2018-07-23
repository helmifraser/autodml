import keras
import keras.backend as K
from keras import optimizers
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils.training_utils import multi_gpu_model

from coloured_print import printc

IMG_WIDTH = 224
IMG_HEIGHT = 224

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True) + 0.0001)

# params are [lr, d_1, d_2, d_3, dr]

def create_model(model_params=[0.001], multi_gpu=False, gpus=2):
    img = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='img')
    mobilenet = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), alpha=1.0,
                            depth_multiplier=1, include_top=False,
                            weights='imagenet', pooling='max')(img)
    # mobilenet.trainable = False

    # Default params
    model = []
    if len(model_params) == 1:
        x = Dense(640, input_shape=(1, 1280), activation='relu')(mobilenet)
        x = Dropout(0.2)(x)
        x = Dense(80, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        steer = Dense(1, activation='tanh', name='steer')(x)
        throttle = Dense(1, activation='tanh', name='throttle')(x)
        # brake = Dense(1, activation='tanh', name='brake')(x)
        model = Model(inputs=img, outputs=[steer, throttle])
    elif len(model_params) == 5:
        print(" ")
        print("Params:")
        print("Learning rate:", end=''); printc(" {}".format(model_params[0]), 'okgreen')
        print("Dense_1:", end=''); printc(" {}".format(model_params[1]), 'okgreen')
        print("Dense_2:", end=''); printc(" {}".format(model_params[2]), 'okgreen')
        print("Dense_3:", end=''); printc(" {}".format(model_params[3]), 'okgreen')
        print("Dropout rate:", end=''); printc(" {}".format(model_params[4]), 'okgreen')
        print(" ")
        x = Dense(model_params[1], input_shape=(1, 1280), activation='relu')(mobilenet)
        x = Dropout(model_params[4])(x)
        x = Dense(model_params[2], activation='relu')(x)
        x = Dropout(model_params[4])(x)
        x = Dense(model_params[3], activation='relu')(x)
        x = Dropout(model_params[4])(x)
        steer = Dense(1, activation='tanh', name='steer')(x)
        throttle = Dense(1, activation='tanh', name='throttle')(x)
        # brake = Dense(1, activation='tanh', name='brake')(x)
        model = Model(inputs=img, outputs=[steer, throttle])
    else:
        printc("Error: malformed model_params argument. Expected size 5 got size {}".format(len(model_params)))

    for layers in model.layers[:2]:
        layers.trainable = False

    adam = optimizers.Adam(lr=model_params[0])

    print("Compiling model, will use ", end='')

    if multi_gpu is True:
        printc("{} GPUs".format(gpus), 'okgreen')
        parallel_model = multi_gpu_model(model, gpus=gpus)
        parallel_model.compile(optimizer=adam, loss=euclidean_distance_loss)
    else:
        printc("a single GPU")

    model.compile(optimizer=adam, loss=euclidean_distance_loss)

    model.summary()

    if multi_gpu is True:
        return parallel_model, model
    else:
        return model
