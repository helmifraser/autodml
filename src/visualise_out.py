import argparse
import numpy as np
import os
import sys
import pandas as pd

from matplotlib import pyplot as plt

from keras.preprocessing.image import img_to_array
from keras.models import Model, load_model
from vis.utils import utils
from vis.visualization import visualize_cam, visualize_saliency, visualize_activation, overlay
from keras.preprocessing import image
from keras import backend as K

from network import euclidean_distance_loss
from network_2 import atan

argparser = argparse.ArgumentParser(
    description='CARLA Deep Model Testing Client')
argparser.add_argument(
    'model', metavar='path',
    type=str,
    help='Path to driving model .h5 file')
argparser.add_argument(
    '-type', metavar='model_type',
    type=int,
    default=0,
    help='Mobilenet based model, or Nvidia based model (0, 1)')

args = argparser.parse_args()


def return_model(model_path, type):
    if os.path.isfile(model_path) is not True:
        print("Model {} not found!")
        sys.exit(0)

    if type == 0:
        model = load_model(model_path, custom_objects={
                           'euclidean_distance_loss': euclidean_distance_loss, 'relu6': K.relu})
    elif type == 1:
        model = load_model(model_path, custom_objects={'atan': atan})
    else:
        print("Model type '{}' not recognised, check argument".format(type))
        sys.exit(0)

    model.summary()

    return model


def obtain_ground_truth(path):
    for filename in os.listdir(path):
        if filename.endswith("_episode.txt"):
            dataframe = pd.read_csv(path+"/"+filename,
                                    delimiter=' ', header=None,
                                    usecols=[1, 2, 3])

    return dataframe


def preprocess_img(img):
    im = image.img_to_array(img)[90:, :, :]
    im = np.expand_dims(im, axis=0)
    im = im*(2./256) - 1
    return im


model = return_model(args.model, args.type)

layer_name = 'dense_1' # 'block1_conv2', 'block2_conv2', 'block3_conv2'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
filter_idx = 0

#     print(layers)
#
# df = obtain_ground_truth("../../carla_dataset/train/0_town_1/")
# # for element in df:
# #     print(df[element])
i = 273
#
img_path = '../../carla_dataset/train/0_town_1/CameraRGB/' + str(i) + '_CameraRGB_0.png'
img = image.load_img(img_path, target_size=(224, 224))
# img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
# grads = visualize_saliency(model, layer_idx, filter_indices=filter_idx, seed_input=img)
# Plot with 'jet' colormap to visualize as a heatmap.

plt.figure()
# plt.imshow(grads, cmap='jet')
plt.imshow(model.layers[layer_idx].get_weights()[0][:, :, :, 0].squeeze(), cmap='gray')# for layers in model.layers:
# plt.imshow(img[..., 0])
plt.show()
# x = preprocess_img(img)
#
# preds = model.predict(x)
#
# print("GT: {} {}, Pred: {}".format(df[1][i], df[2][i], preds))
#
# titles = ["left", "right", "straight"]
# modifiers = [None, 'negate', 'small_values']
# for i, modifier in enumerate(modifiers):
#     heatmap = visualize_cam(model, layer_idx=layer_idx, filter_indices=0,
#                             seed_input=img, grad_modifier=modifier)
#     plt.figure()
#     plt.title(titles[i])
#
#     # Overlay is used to alpha blend heatmap onto img.
#     plt.imshow(overlay(img, heatmap, alpha=0.7))
