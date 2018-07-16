import os
import subprocess as sp
import numpy as np
import pandas as pd
import keras
import random
from livelossplot import PlotLossesKeras

from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, load_model, Sequential
from keras.layers import Flatten, Dense, Dropout, Input
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers
from scipy.misc import imsave

np.random.seed(420)  # for high reproducibility

DATASET_PATH = "carla_dataset/test"
DATA_HEADERS = ["frame_no", "steer", "throttle", "brake", "reverse"]
COLS_TO_USE = [1, 2, 3]
IMG_WIDTH = 224
IMG_HEIGHT = 224

def create_model(learning_rate=0.01):
    img = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3), name='img')
    mobilenet = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', pooling='max')(img)
    x = Dense(64, input_shape=(1, 1280), activation='relu')(mobilenet)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    steer = Dense(1, activation='tanh', name='steer')(x)
    throttle = Dense(1, activation='tanh', name='throttle')(x)
    brake = Dense(1, activation='tanh', name='brake')(x)
    model = Model(inputs=img, outputs=[steer, throttle, brake])

    # print("All layers added")

    adam = optimizers.Adam(lr=learning_rate)

    # print("Compiling")

    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
    model.summary()
    return model

def obtain_data(data_path):
    """Returns all data in folder. Path is top level of data i.e ../dataset/train"""
    folders = os.listdir(data_path)
    x_data = [None] * len(folders)
    y_data = [None] * len(folders)

    print("Within '{}', I found {} folders: {}".format(data_path, len(folders), folders))

    for idx, folder in enumerate(folders):
        print("Fetching episode {} of {}: {}".format(idx, len(folders) - 1, folder))
        x_data[idx], y_data[idx] = obtain_episode_data(data_path+"/"+folder, header_names=DATA_HEADERS)

    print("All done")
    return x_data, y_data

def obtain_episode_data(folder_path, seg=False, delim=' ', header_names=None,
                        columns_to_use=COLS_TO_USE):
    # Obtain images i.e x data
    rgb_path = folder_path+"/CameraRGB"
    rgb_images = []
    seg_images = []

    if os.path.isdir(rgb_path):
        rgb_images = image_extract(rgb_path)
    else:
        print("Error:- folder '{}' not found".format(rgb_path))

    if seg == True:
        seg_path = folder_path+"/CameraSemSeg"
        if os.path.isdir(seg_path):
            # files_seg = sum(os.path.isdir(i) for i in os.listdir(seg_path))
            seg_images = image_extract(seg_path)
        else:
            print("Error:- folder '{}' not found".format(seg_path))

    # Obtain y data

    for filename in os.listdir(folder_path):
        if filename.endswith("_episode.txt"):
            dataframe = pd.read_csv(folder_path+"/"+filename,
                                    delimiter=delim, header=None,
                                    usecols=columns_to_use)

    dataset = dataframe.values
    control_data = dataset[:].astype(float)
    return rgb_images, control_data

def image_extract(folder_path, number_of_images=800):
        images = [None] * number_of_images

        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                pos = filename.find("_")
                id = int(filename[:pos])
                img = load_img(os.path.join(folder_path, filename), target_size=(224, 224))
                images[id] = img_to_array(img)
                # images[id] = np.expand_dims(images[id], axis=0)
                images[id] = preprocess_input(images[id])
                # print(np.shape(images[id]))
                # print("Shape: {}".format(np.shape(images[1])))

        images = list(filter(None.__ne__, images))
        return images

def batch_generator(features, labels, batch_size=32):
    # Create empty arrays to contain batch of features and labels
    # batch_features = np.zeros((batch_size, IMG_WIDTH, IMG_HEIGHT, 3))
    # batch_labels = np.zeros((batch_size,1))

    batch_features = [None] * batch_size
    batch_labels = [None] * batch_size

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = np.random.randint(0, len(features), 1)
            batch_features[i] = features[index[0]]
            batch_labels[i] = labels[index[0]]

        batch_labels = np.asarray(batch_labels)
        yield batch_features, batch_labels


train_datagen = ImageDataGenerator(
        rotation_range=45,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

# x_train, y_train = obtain_data(DATASET_PATH)
# print("x: {}, y: {}".format(np.shape(x_train), np.shape(y_train)))

model = create_model()

# tensorboard_cb = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0,
#           write_graph=True, write_images=True)

# for episode in range(1):
#     print('Episode', episode)
#     for epoch in range(5):
#         for img in range(600):
#             print("Epoch {} Image: {}".format(epoch, img))
#             model.fit(x_train[episode][img],
#                     [ [y_train[episode][img][0]],
#                     [y_train[episode][img][1]],
#                     [y_train[episode][img][2]] ],
#                     # callbacks=[tensorboard_cb],
#                     verbose=0)


def train(data_path, num_epochs=50):
    folders = os.listdir(data_path)

    for idx, folder in enumerate(folders):
        x_data, y_data = obtain_episode_data(data_path+"/"+folder, header_names=DATA_HEADERS)

        for epoch in range(num_epochs):
            print("Epoch", epoch)

            batches = 0
            for batch, labels in batch_generator(x_data, y_data):
                print("Episode {}/{}, epoch {}/{}, batch {}/{}".format(idx, len(folders),
                                                            epoch, num_epochs,
                                                            batches,
                                                            32))
                model.fit([batch], [labels[...,0], labels[...,1], labels[...,2]])
                batches += 1
                if batches >= len(x_data) / 32:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
                # tmp = sp.call('clear', shell=True)

    model.save_weights('test.h5')

train(DATASET_PATH)
# x_train, y_train = obtain_episode_data("carla_dataset/train/0_town_1", headers=DATA_HEADERS)


# here's a more "manual" example
# for e in range(1, 11):
#     print('Epoch', e)
#     batches = 0
#     for x_batch in datagen.flow(x_train, batch_size=32):
#         model.fit(x_batch, y_train)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break
