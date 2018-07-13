import os
import numpy as np
import pandas as pd
import keras
import random

from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, load_model
from keras import optimizers
from scipy.misc import imsave

np.random.seed(420)  # for reproducibility
DATA_HEADERS = ["frame_no", "steer", "throttle", "brake", "reverse"]
COLS_TO_USE = [1, 2, 3]
IMG_WIDTH = 224
IMG_HEIGHT = 224

def create_model(learning_rate=0.01):
    model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', pooling='max')
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='relu'))

    adam = optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    return model

def obtain_episode_data(folder_path, seg=False, delim=' ', headers=None,
                        use_index=False, col_index=0, columns_to_use=COLS_TO_USE):
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
            if use_index == True:
                dataframe = pd.read_csv(folder_path+"/"+filename,
                                        delimiter=delim, names=headers,
                                        index_col=col_index, usecols=columns_to_use)
            else:
                dataframe = pd.read_csv(folder_path+"/"+filename,
                                        delimiter=delim,
                                        names=headers, usecols=columns_to_use)

    dataset = dataframe.values
    control_data = dataset[:].astype(float)
    return rgb_images, control_data

def image_extract(folder_path, number_of_images=600):
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
        yield batch_features, batch_labels


train_datagen = ImageDataGenerator(
        rotation_range=45,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

x_train, y_train = obtain_episode_data("carla_dataset/train/0_town_1", headers=DATA_HEADERS)

mobilenet_v2 = create_model()

# print("Shape x_train: {}".format(np.shape(x_train)))
# print("Shape y_train: {}".format(np.shape(y_train)))

# def return_batch(x_data, y_data, batch_size=32):
#     i = 0
#     for batch, labels in batch_generator(x_train, y_train, batch_size):
#         i += 1
#         print("{} Shape batch: {}".format(i, np.shape(batch)))
#         print("{} Shape labels: {}".format(i, np.shape(labels)))
#         print(labels)
#         if i > 2:
#             break  # otherwise the generator would loop indefinitely

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
