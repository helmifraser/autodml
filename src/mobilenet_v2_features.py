import os
import numpy as np
import pandas as pd
import keras

from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from scipy.misc import imsave

np.random.seed(420)  # for reproducibility
DATA_HEADERS = ["frame_no", "steer", "throttle", "brake", "reverse"]

def obtain_episode_data(folder_path, seg=False, delim=' ', headers=None):
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
            dataframe = pd.read_csv(folder_path+"/"+filename, delimiter=delim, names=headers, index_col=0)

    dataset = dataframe.values
    control_data = dataset[:].astype(float)

    return rgb_images, control_data

def image_extract(folder_path, number_of_images=600):
        images = [None] * number_of_images

        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                pos = filename.find("_")
                id = int(filename[:pos])
                img = image.load_img(os.path.join(folder_path, filename), target_size=(224, 224))
                images[id] = image.img_to_array(img)
                images[id] = np.expand_dims(images[id], axis=0)
                images[id] = preprocess_input(images[id])
                # print(shape(rgb_images[id]))
                # print("Shape: {}".format(np.shape(rgb_images[1])))

        return images

x_train, y_train = obtain_episode_data("carla_dataset/train/0_town_1", headers=DATA_HEADERS)

print("Shape x_train: {}".format(np.shape(x_train)))
print("Shape y_train: {}".format(np.shape(y_train)))


# Removes the top and displays only features

# model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', pooling='avg')

# img_path = 'elephant.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# print("Shape x: {}".format(np.shape(x)))
#
# features = model.predict(x)
#
# # model.summary()
#
# print("Shape: {}".format(np.shape(features)))
