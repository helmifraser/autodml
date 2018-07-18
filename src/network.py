import keras
from keras import optimizers
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


IMG_WIDTH = 224
IMG_HEIGHT = 224


def create_model(learning_rate=0.001):
    img = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='img')
    mobilenet = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), alpha=1.0,
                            depth_multiplier=1, include_top=False,
                            weights='imagenet', pooling='max')(img)
    # mobilenet.trainable = False
    x = Dense(640, input_shape=(1, 1280), activation='relu')(mobilenet)
    x = Dropout(0.2)(x)
    x = Dense(80, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    steer = Dense(1, activation='tanh', name='steer')(x)
    throttle = Dense(1, activation='relu', name='throttle')(x)
    brake = Dense(1, activation='relu', name='brake')(x)
    model = Model(inputs=img, outputs=[steer, throttle, brake])

    for layers in model.layers[:2]:
        layers.trainable = False

    # print("All layers added")

    adam = optimizers.Adam(lr=learning_rate)

    # print("Compiling")

    model.compile(optimizer=adam, loss='mean_squared_error',
                  metrics=['mse'])
    model.summary()
    return model
