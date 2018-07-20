import os
import numpy as np
import pandas as pd
from datetime import datetime

import keras
from keras import optimizers
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# from scipy.misc import imsave

from loss_history import LossHistory
import network

np.random.seed(420)  # for high reproducibility

os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
USE_MULTI=True
DATASET_PATH = "../../carla_dataset/train"
DATA_HEADERS = ["frame_no", "steer", "throttle", "brake", "reverse"]
COLS_TO_USE = [1, 2, 3]


def obtain_data(data_path):
    """Returns all data in folder. Path is top level of data i.e ../dataset/train"""
    folders = os.listdir(data_path)
    x_data = [None] * len(folders)
    y_data = [None] * len(folders)

    print("Within '{}', I found {} folders: {}".format(
        data_path, len(folders), folders))

    for idx, folder in enumerate(folders):
        print("Fetching episode {} of {}: {}".format(
            idx, len(folders) - 1, folder))
        x_data[idx], y_data[idx] = obtain_episode_data(
            data_path+"/"+folder, header_names=DATA_HEADERS)

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
                img = load_img(os.path.join(
                    folder_path, filename), target_size=(224, 224))
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

def digit_counter(s):
    return [sum(n) for n in zip(*((c.isdigit(), c.isalpha()) for c in s))]

def train(data_path, model, callbacks, target_model_name, num_epochs=50,
          number_of_batches=32, batch_size=32, test=False):
    folders = os.listdir(data_path)
    checkpoint = 0
    for idx, folder in enumerate(folders):
        x_data, y_data = obtain_episode_data(
            data_path+"/"+folder, header_names=DATA_HEADERS)

        for epoch in range(num_epochs):
            print("Epoch", epoch)

            batch_count = 0
            for batch, labels in batch_generator(x_data, y_data, batch_size=batch_size):
                print("Episode {}/{}, epoch {}/{}, batch {}/{}".format(idx,
                                                                       len(folders)-1,
                                                                       epoch, num_epochs,
                                                                       batch_count,
                                                                       number_of_batches))
                model.fit([batch], [labels[..., 0],
                                    labels[..., 1],
                                    labels[..., 2]],
                          callbacks=callbacks)
                batch_count += 1
                if batch_count >= number_of_batches:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break

        checkpoint += 1
        if checkpoint == 3:
            print("Checkpoint: Saving model to "
                  + "../weights/checkpoints/checkpoint_"
                  + str(idx) + "_" + target_model_name + '.h5')
            model.save("../weights/checkpoints/checkpoint_" + str(idx) + "_"
                       + target_model_name + '.h5')
            checkpoint = 0

        if test is True:
            print("Testing only, breaking")
            break

    print("Saving model to "+"../weights/"+target_model_name+'.h5')
    model.save("../weights/"+target_model_name+'.h5')


def train_with_all(data_path, model, target_model_name, nb_epochs=10,
                   checkpoint_stage=10, callbacks=None, save_model=None):
    folders = os.listdir(data_path)
    checkpoint = 0
    history_file = open("../weights/histories/history_file_"
    + target_model_name+".txt", 'a')
    for idx, folder in enumerate(folders):
        print("Obtaining data: {}/{}".format(idx, len(folders) - 1))
        x_data, y_data = obtain_episode_data(
            data_path+"/"+folder, header_names=DATA_HEADERS)
        x_data = np.asarray(x_data)
        y_data = np.asarray(y_data)

        history = model.fit([x_data], [y_data[..., 0],
                                       y_data[..., 1]],
                            epochs=nb_epochs, callbacks=callbacks)

        checkpoint += 1
        history_file.write(str(history.history['loss'])+"\n")
        if checkpoint == checkpoint_stage:
            print("Checkpoint: Saving model to " + "../weights/checkpoints/checkpoint_"
                  + str(idx) + "_" + target_model_name + '.h5')
            if save_model is not None:
                save_model.save("../weights/checkpoints/checkpoint_" + str(idx) + "_"
                           + target_model_name + '.h5')
            checkpoint = 0

    print("Saving model to "+"../weights/twa_final_"+target_model_name+'.h5')
    save_model.save("../weights/twa_final_"+target_model_name+'.h5')

gpus = digit_counter(os.environ["CUDA_VISIBLE_DEVICES"])[0]
name = str(datetime.now()).replace(" ", "_")

parallel_model, model = network.create_model(multi_gpu=USE_MULTI, gpus=gpus)

# tensorboard_cb = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0,
#           write_graph=True, write_images=True)
stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=1,
                     mode='auto')
# checkpointer = ModelCheckpoint(filepath='../weights/checkpoint.hdf5', verbose=1,
#                                save_best_only=True)
history = LossHistory()

callbacks = [history, stop]

train_with_all(DATASET_PATH,
                target_model_name=name,
                model=parallel_model,
                save_model=model,
                nb_epochs=20,
                callbacks=callbacks)

print("All done")
