import keras
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model

import numpy as np
from scipy.misc import imsave

# Removes the top and displays only features

model = MobileNetV2(input_shape=(224, 224, 3), alpha=1.0, depth_multiplier=1, include_top=False, weights='imagenet', pooling='avg')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

# model.summary()

print("Shape: {}".format(np.shape(features)))
