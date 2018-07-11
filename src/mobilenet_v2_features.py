import os
import numpy as np
import pandas as pd
import keras

from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model, load_model
from scipy.misc import imsave

np.random.seed(420)  # for reproducibility
DATA_HEADERS = ["frame_no", "steer", "throttle", "brake", "reverse"]
COLS_TO_USE = [1, 2, 3]

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
                images[id] = np.expand_dims(images[id], axis=0)
                images[id] = preprocess_input(images[id])
                # print(shape(images[id]))
                # print("Shape: {}".format(np.shape(images[1])))

        return images

datagen = ImageDataGenerator(
        rotation_range=45,
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

x_train, y_train = obtain_episode_data("carla_dataset/train/0_town_1", headers=DATA_HEADERS)

# x_batch = datagen.flow(x_train, batch_size=1)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x_train[0], batch_size=1,
#                           save_to_dir='preview', save_prefix='test', save_format='png'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely

print("Shape x_train: {}".format(np.shape(x_train)))
print("Shape y_train: {}".format(np.shape(y_train)))
# print("Shape x_batch: {}".format(np.shape(x_batch)))

# here's a more "manual" example
# for e in range(1, 11):
#     print('Epoch', e)
    # batches = 0
    # for x_batch in datagen.flow(x_train, batch_size=32):
    #     model.fit(x_batch, y_train)
    #     batches += 1
    #     if batches >= len(x_train) / 32:
    #         # we need to break the loop by hand because
    #         # the generator loops indefinitely
    #         break
