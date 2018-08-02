import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
import argparse
import random


import keras
from keras import optimizers
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import backend as K


import scipy
from scipy.misc import imsave, pilutil
from skimage.transform import rotate
from skimage.util import random_noise

from loss_history import LossHistory
import network
import network_2
from coloured_print import printc

np.random.seed(420)  # for high reproducibility

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument('d', metavar='dataset_path', type=str,
                    help='Path to training set')
parser.add_argument('v', metavar='valset_path', type=str,
                    help='Path to validation set')
parser.add_argument('-gpus', metavar='gpu_number(s)', type=int, nargs='+',
                    default=[0], help='GPUs visible to this script')

args = parser.parse_args()
gpu_number = args.gpus
d_path = args.d
v_path = args.v

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)[1:-1]

import tensorflow as tf
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.75
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(session)

USE_MULTI = False
if len(gpu_number) > 1:
    USE_MULTI = True

DATASET_PATH = d_path
VALSET_PATH = v_path
DATA_HEADERS = ["frame_no", "steer", "throttle", "brake", "reverse"]
COLS_TO_USE = [1, 2, 3]


def obtain_data(data_path, seg=False):
    """Returns all data in folder. Path is top level of data i.e ../dataset/train"""
    folders = os.listdir(data_path)
    random.shuffle(folders)
    # x_data = [None] * len(folders)
    # y_data = [None] * len(folders)
    # limit = len(folders)
    limit = 1

    # x_data = np.zeros((570*(limit+1), 224, 224, 3))
    # x_seg = np.zeros((570*(limit+1), 224, 224, 3))
    # y_data = np.zeros((570*(limit+1), 3))
    x_data = []
    x_seg = []
    y_data = []

    print("Within '{}', I found {} folders: {}".format(
        data_path, len(folders), folders))
    for idx, folder in enumerate(folders):
        if idx > limit:
            break

        print("Fetching episode {} of {}: {}".format(
            idx, limit, folder))
        if seg:
            x_ep, x_ep_seg, y_ep = obtain_episode_data(
                data_path+"/"+folder, seg=seg, header_names=DATA_HEADERS)
        else:
            x_ep, y_ep = obtain_episode_data(
                data_path+"/"+folder, header_names=DATA_HEADERS)
        samples = np.shape(y_ep)[0]
        x_data[idx*samples:samples + (idx*samples)] = x_ep
        y_data[idx*samples:samples + (idx*samples)] = y_ep

        if seg is True:
            x_seg[idx*samples:samples + (idx*samples)] = x_ep_seg

    print("All done")

    if seg is True:
        return np.asarray(x_data), \
                np.asarray(x_seg), \
                np.asarray(y_data)

    return np.asarray(x_data), np.asarray(y_data)


def filter_control_data(data):
    # get rid of frames where the car has stopped
    idx = np.where(data[:, 1] > 0.1)
    valid_data = data[idx]
    idx = [i for i in idx[0]]
    return valid_data, idx


def return_straight_ids(data, thresh=0.003):
    return np.where(abs(data[:, 0]) < thresh)


def return_turn_ids(data, thresh=0.035):
    return np.where(abs(data[:, 0]) > thresh)


def strip_straight(x_data, y_data, rate=0.25):
    turns_idx = return_straight_ids(y_data)
    # print(np.shape(turns_idx))
    # print(turns_idx)

    if len(turns_idx[0]) > 2:
        np.random.shuffle(turns_idx[0])
        # print(turns_idx)
        choice = int(len(turns_idx[0])*rate)

        ordered = []
        for counter, id in enumerate(turns_idx[0]):
            if counter >= choice:
                break
            ordered.append(id)

        ordered.sort(reverse=True)

        for id in ordered:
            # print(id)
            x_data = np.delete(x_data, id, 0)
            y_data = np.delete(y_data, id, 0)

    return x_data, y_data
    # x = []
    # y = []

    # for counter, id in enumerate(turns_idx[0]):
    #     if counter >= int(len(turns_idx[0])*rate):
    #         break
    #
    #     x.append(x_data[id])
    #     y.append(y_data[id])
    #
    # return np.asarray(x), np.asarray(y)

def obtain_episode_data(folder_path, seg=False, delim=' ', header_names=None,
                        columns_to_use=COLS_TO_USE):
    rgb_path = folder_path+"/CameraRGB"
    seg_path = folder_path+"/CameraSemSeg"
    rgb_images = []
    seg_images = []

    # Obtain y data

    for filename in os.listdir(folder_path):
        if filename.endswith("_episode.txt"):
            dataframe = pd.read_csv(folder_path+"/"+filename,
                                    delimiter=delim, header=None,
                                    usecols=columns_to_use)

    dataset = dataframe.values
    control_data = dataset[:].astype(float)
    control_data, idx = filter_control_data(control_data)
    turns_idx = return_turn_ids(control_data)

    # Obtain images i.e x data

    if os.path.isdir(rgb_path):
        rgb_images = np.asarray(image_extract(rgb_path))
        rgb_images = rgb_images[idx, ]
    else:
        printc("Error:- folder '{}' not found".format(rgb_path))

    if seg == True:
        if os.path.isdir(seg_path):
            seg_images = np.asarray(image_extract(seg_path, seg=seg))
            seg_images = seg_images[idx, ]
        else:
            printc("Error:- folder '{}' not found".format(seg_path))

    if seg == True:
        return np.asarray(rgb_images[30:]), \
                np.asarray(seg_images[30:]), \
                np.asarray(control_data[30:])

    return np.asarray(rgb_images), np.asarray(control_data)


def image_extract(folder_path, seg=False, number_of_images=800):
        images = [None] * number_of_images

        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                pos = filename.find("_")
                id = int(filename[:pos])
                img = load_img(os.path.join(
                    folder_path, filename), target_size=(224, 224))
                images[id] = img_to_array(img)

                # images[id] = img_to_array(img)[90:, :, :]
                # print("Before {}".format(images[id]))

                # images[id] = np.expand_dims(images[id], axis=0)
                if seg is False:
                    images[id] = preprocess_input(images[id])
                else:
                    images[id] = (images[id]*2./12) - 1

                # print(np.shape(images[id]))
                # print("Shape: {}".format(np.shape(images[1])))

        images = list(filter(None.__ne__, images))
        return images


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    # scipy.ndimage.interpolation.rotate(image_array, random_degree, mode='nearest')
    # return rotate(image_array, random_degree)
    return scipy.ndimage.interpolation.rotate(image_array, random_degree, reshape=False, mode='nearest')


def add_random_noise(image_array):
    # add random noise to the image
    return random_noise(image_array)


def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return np.fliplr(image_array)


def augment_brightness(image_array, factor=.75):
    random_bright = factor+np.random.uniform(-1, 1)
    return image_array*random_bright


def augment_episode(x_data, y_data, x_seg=None):

    available_transformations = {
        'noise': add_random_noise,
        'horizontal_flip': horizontal_flip,
        'brightness': augment_brightness
    }

    x_aug = []
    y_aug = []
    x_seg_aug = []

    for idx, image in enumerate(x_data):
        num_transformations_to_apply = random.randint(
            0, len(available_transformations))
        for transformations in range(num_transformations_to_apply):
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            # print(key)
            x_aug.append(available_transformations[key](image))
            y_aug.append(y_data[idx])

            if x_seg is not None:
                x_seg_aug.append(available_transformations[key](x_seg[idx]))

            if key is 'horizontal_flip':
                y_aug[-1][0] = -1*y_aug[-1][0]

    if x_seg is not None:
        return x_aug, x_seg_aug, y_aug

    return x_aug, y_aug


def augment_turns(x_data, y_data, x_seg=None, straight=False):
    available_transformations = {
        'noise': add_random_noise,
        'brightness': augment_brightness
    }

    x_aug = []
    y_aug = []
    x_seg_aug = []

    # print(turns_idx)
    turns_idx = return_turn_ids(y_data)
    if straight is True:
        turns_idx = return_straight_ids(y_data)
        np.random.shuffle(turns_idx[0])
        tmp = turns_idx[0]
        # print(np.shape(turns_idx))
        choice = int(len(tmp)*0.5)
        tmp = np.delete(tmp, np.s_[0:choice])
        turns_idx = [tmp]

    for idx in turns_idx[0]:
        flipped = horizontal_flip(x_data[idx])
        x_aug.append(flipped)
        y_aug.append(y_data[idx])
        y_aug[-1][0] = -1*y_aug[-1][0]

        if x_seg is not None:
            flipped_seg = horizontal_flip(x_seg[idx])
            x_seg_aug.append(flipped_seg)

        num_transformations_to_apply = random.randint(
            0, len(available_transformations))
        for transformations in range(num_transformations_to_apply):
            # choose a random transformation to apply for a single image
            key = random.choice(list(available_transformations))
        #     # print(key)
            x_aug.append(available_transformations[key](x_data[idx]))
            y_aug.append(y_data[idx])
            x_aug.append(available_transformations[key](flipped))
            y_aug.append(y_data[idx])
            y_aug[-1][0] = -1*y_aug[-1][0]

            if x_seg is not None:
                x_seg_aug.append(available_transformations[key](x_seg[idx]))
                x_seg_aug.append(available_transformations[key](flipped_seg))

    if x_seg is not None:
        return x_aug, x_seg_aug, y_aug

    return x_aug, y_aug


def batch_generator(features, labels, batch_size=32):
    # Create empty arrays to contain batch of features and labels
    # batch_features = np.zeros((batch_size, IMG_WIDTH, IMG_HEIGHT, 3))
    # batch_labels = np.zeros((batch_size,1))

    batch_features = [None] * batch_size
    batch_labels = [None] * batch_size

    while True:
        for i in range(batch_size):
            num_transformations_to_apply = random.randint(
                1, len(available_transformations))
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
            model.save("../weighlen(folders)ts/checkpoints/checkpoint_" + str(idx) + "_"
                       + target_model_name + '.h5')
            checkpoint = 0

        if test is True:
            print("Testing only, breaking")
            break

    print("Saving model to "+"../weights/"+target_model_name+'.h5')
    model.save("../weights/"+target_model_name+'.h5')


def train_with_all(data_path, val_path, model, target_model_name, nb_epochs=10,
                   checkpoint_stage=10, callbacks=None, save_model=None):
    # folders = os.listdir(data_path)
    # # random.shuffle(folders)
    #
    # # Testing overfitting, train on same episode for all
    # folders.sort()
    # # print(folders)
    #
    # t = 1

    os.mkdir("../weights/"+target_model_name)
    history_file, val_history_file = create_history_files(target_model_name)

    # print("Training on {}".format(folders[t]))
    # x_data, x_seg, y_data = obtain_episode_data(
    #     data_path+"/"+folders[t], seg=True, header_names=DATA_HEADERS)
    #
    # # x_aug, x_seg_aug, y_aug = augment_episode(x_data, y_data, x_seg)
    # x_aug, x_seg_aug, y_aug = augment_turns(x_data, y_data, x_seg)
    #
    # if np.shape(x_aug)[0] > 0 :
    #     x_data = np.append(x_data, x_aug, axis=0)
    #     x_seg = np.append(x_seg, x_seg_aug, axis=0)
    #     y_data = np.append(y_data, y_aug, axis=0)
    #
    # history = fit_over_ep(model, target_model_name, nb_epochs, callbacks, [x_data, x_seg, y_data],
    #                       val_path, 5, history_file, val_history_file)
    #
    # print("Training on {}".format(folders[t+23]))
    # x_data, x_seg, y_data = obtain_episode_data(
    #     data_path+"/"+folders[t+23], seg=True, header_names=DATA_HEADERS)
    #
    # x_aug, x_seg_aug, y_aug = augment_turns(x_data, y_data, x_seg)
    #
    # x_data = np.append(x_data, x_aug, axis=0)
    # x_seg = np.append(x_seg, x_seg_aug, axis=0)
    # y_data = np.append(y_data, y_aug, axis=0)
    #
    # history = fit_over_ep(model, target_model_name, nb_epochs, callbacks, [x_data, x_seg, y_data],
    #                       val_path, 5, history_file, val_history_file)

    history = fit_over_folders(data_path, model, model, target_model_name,
                               nb_epochs, callbacks, val_path,
                               history_file, val_history_file)

    printc("Saving model to "+"../weights/"+target_model_name+"/twa_final_" +
           target_model_name+'.h5', 'okgreen')
    save_model.save("../weights/"+target_model_name +
                    "/twa_final_"+target_model_name+'.h5')


def create_history_files(target_model_name):
    history_file = open("../weights/"+target_model_name+"/history_file_"
                        + target_model_name+".txt", 'a')
    val_history_file = open("../weights/"+target_model_name+"/val_history_file_"
                            + target_model_name+".txt", 'a')

    return history_file, val_history_file


def fit_over_ep(model, target_model_name, nb_epochs, callbacks, episode_data, val_path, iterations, history_file, val_history_file):
    checkpoint = 0
    history = []

    x_data = episode_data[0]
    x_seg = episode_data[1]
    y_data = episode_data[2]

    for i in range(iterations):
        # x_val, x_val_seg, y_val = obtain_data(val_path, seg=True)
        history = model.fit([x_data, x_seg],
                            [y_data[..., 0],
                             y_data[..., 1]],
                            # validation_data=([x_val, x_val_seg],
                            #                  [y_val[..., 0],
                            #                   y_val[..., 1]]),
                            epochs=nb_epochs,
                            callbacks=callbacks,
                            )
        checkpoint += 1
        history_file.write(str(history.history['loss'])+"\n")
        # val_history_file.write(str(history.history['val_loss'])+"\n")
        if checkpoint == 1:
            printc("Checkpoint: Saving model to " + "../weights/"+target_model_name+"/checkpoint_"
                   + str(i) + "_" + target_model_name + '.h5', 'okgreen')
            if model is not None:
                model.save("../weights/"+target_model_name+"/checkpoint_" + str(i) + "_"
                           + target_model_name + '.h5')
            checkpoint = 0

    return history


def fit_over_folders(data_path, model, save_model, target_model_name,
                     nb_epochs, callbacks, val_path,
                     history_file, val_history_file, checkpoint_stage=10):

    folders = os.listdir(data_path)
    random.shuffle(folders)
    checkpoint = 0
    history = []

    x_data = []
    y_data = []
    x_seg = []

    # folders.sort()
    x_val, y_val = obtain_data(val_path, seg=False)

    for idx, folder in enumerate(folders):
        # if (folder != '72_town_1') and (folder != '73_town_1'):
        #     continue

        # if (folder != '73_town_1'):
        #     continue

        if idx >= 10:
            break
        for i in range(1):
            if i == 0:
                print("Obtaining data: {} {}/{}".format(folder, idx, len(folders) - 1))
                x_data, y_data = obtain_episode_data(
                    data_path+"/"+folder, header_names=DATA_HEADERS)

                # print(np.shape(y_data))
                # x_data, y_data = strip_straight(x_data, y_data)
                # print(np.shape(y_data))

                x_aug, y_aug = augment_turns(x_data, y_data)

                if np.shape(x_aug)[0] > 0:
                    x_data = np.append(x_data, x_aug, axis=0)
                    # x_seg = np.append(x_seg, x_seg_aug, axis=0)
                    y_data = np.append(y_data, y_aug, axis=0)

                x_aug, y_aug = augment_turns(x_data, y_data, straight=True)
                if np.shape(x_aug)[0] > 0:
                    x_data = np.append(x_data, x_aug, axis=0)
                    # x_seg = np.append(x_seg, x_seg_aug, axis=0)
                    y_data = np.append(y_data, y_aug, axis=0)

            history = model.fit([x_data],
                                [y_data[..., 0],
                                 y_data[..., 1]],
                                validation_data=([x_val],
                                                 [y_val[..., 0],
                                                  y_val[..., 1]]),
                                epochs=nb_epochs,
                                callbacks=callbacks,
                                )
            checkpoint += 1
            history_file.write(str(history.history['loss'])+"\n")
            val_history_file.write(str(history.history['val_loss'])+"\n")
            if checkpoint == checkpoint_stage:
                printc("Checkpoint: Saving model to " + "../weights/"+target_model_name+"/checkpoint_"
                       + str(i) + "_" + target_model_name + '.h5', 'okgreen')
                if save_model is not None:
                    save_model.save("../weights/"+target_model_name+"/checkpoint_" + str(idx) + "_"
                                    + target_model_name + '.h5')
                checkpoint = 0

    return history


def main():
    gpus = digit_counter(os.environ["CUDA_VISIBLE_DEVICES"])[0]

    params = [
        [0.001, 100, 100, 0.05],
        [0.001, 100, 100, 0.05],
        [0.001, 100, 100, 0.05],
        [0.001, 100, 100, 0.05],
        [0.001, 100, 100, 0.05]]
        # [0.001, 100, 100, 0.1],
        # [0.001, 100, 100, 0.3],
        # [0.001, 100, 100, 0.8],
        # [0.001, 200, 50, 0.01],
        # [0.001, 200, 50, 0.1],
        # [0.001, 200, 50, 0.3],
        # [0.001, 200, 50, 0.8]]

    for id, param in enumerate(params):
        name = str(param).replace(" ", "_") + \
                   str(datetime.now()).replace(" ", "_").replace("[", "").replace("]", "-")

        # tensorboard_cb = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0,
        #           write_graph=True, write_images=True)
        stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1,
                             mode='auto')
        # checkpointer = ModelCheckpoint(filepath='../weights/checkpoint.hdf5', verbose=1,
        #                                save_best_only=True)
        history = LossHistory()

        callbacks = [history, stop]

        if USE_MULTI is True:
            parallel_model, model = network.create_model(
                            model_params=param,
                            seg=False,
                            multi_gpu=USE_MULTI,
                            gpus=gpus)
            train_with_all(DATASET_PATH, VALSET_PATH,
                           target_model_name=name,
                           model=parallel_model,
                           save_model=model,
                           nb_epochs=200,
                           callbacks=callbacks)

        else:
            model = network.create_model(
                            model_params=param,
                            seg=False,
                            multi_gpu=USE_MULTI,
                            gpus=gpus)
            # model = network_2.create_model(param)

            train_with_all(DATASET_PATH, VALSET_PATH,
                           target_model_name=name,
                           model=model,
                           save_model=model,
                           nb_epochs=50,
                           callbacks=callbacks,
                           checkpoint_stage=10)

            printc("Param set: {} done".format(id))
            printc("Params: {}".format(param))

            printc("All done", 'okgreen')


if __name__ == '__main__':
    main()
