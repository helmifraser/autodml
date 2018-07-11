import keras
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model

import numpy as np

# Performs classification on the specified image
# Spits out preditions

model = MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)

img_path = 'cheetah.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

model.summary()

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)

print('Predicted:', decode_predictions(preds, top=3)[0])
