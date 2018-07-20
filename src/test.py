import os
import sys
import numpy as np
import pandas as pd
import keras

from keras.models import load_model
from keras.applications import mobilenetv2
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.utils.generic_utils import CustomObjectScope
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
from datetime import datetime

from coloured_print import printc

np.random.seed(420)  # for high reproducibility

os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
USE_MULTI=True

MODEL_PATH = "../../results/640-80-16-0.7-dropout/640-80-16-0.7-dropout.h5"
DATASET_PATH = "../../carla_dataset/val"
DATA_HEADERS = ["frame_no", "steer", "throttle", "brake", "reverse"]
COLS_TO_USE = [1, 2, 3]


def obtain_episode_data(folder_path, seg=False, delim=' ', header_names=None,
                        columns_to_use=COLS_TO_USE):
    # Obtain images i.e x data
    rgb_path = folder_path+"/CameraRGB"
    rgb_images = []
    seg_images = []

    if os.path.isdir(rgb_path):
        rgb_images = image_extract(rgb_path)
    else:
        printc("Error:- folder '{}' not found".format(rgb_path))

    if seg == True:
        seg_path = folder_path+"/CameraSemSeg"
        if os.path.isdir(seg_path):
            # files_seg = sum(os.path.isdir(i) for i in os.listdir(seg_path))
            seg_images = image_extract(seg_path)
        else:
            printc("Error:- folder '{}' not found".format(seg_path))

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
                img = load_img(os.path.join(
                    folder_path, filename), target_size=(224, 224))
                images[id] = img_to_array(img)
                # images[id] = np.expand_dims(images[id], axis=0)
                images[id] = preprocess_input(images[id])
                # print(np.shape(images[id]))
                # print("Shape: {}".format(np.shape(images[1])))

        images = list(filter(None.__ne__, images))
        return images

def test(model_path, data_path, callbacks=None):
    if os.path.isfile(model_path) is not True:
        printc("Model {} not found!")
        sys.exit(0)

    folders = os.listdir(data_path)
    n_episodes = len(folders)

    printc("Loading model: {}".format(model_path[model_path.rfind("/")+1:]), 'warn')
    model = load_model(model_path, custom_objects={'relu6': K.relu})
    model.summary()

    printc("Episodes in dataset: {}".format(n_episodes), 'okgreen')

    name = str(datetime.now()).replace(" ", "_")
    history_file = open("../weights/histories/val_history_file_"
    + name+".txt", 'a')

    for idx, folder in enumerate(folders):
        print("Obtaining data: {}/{}".format(idx, len(folders) - 1))
        x_data, y_data = obtain_episode_data(
            data_path+"/"+folder, header_names=DATA_HEADERS)
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        score = model.evaluate([x_data], [y_data[..., 0], y_data[..., 1]])
        history_file.write(str(score)+"\n")

    history_file.close()

test(model_path=MODEL_PATH, data_path=DATASET_PATH)
