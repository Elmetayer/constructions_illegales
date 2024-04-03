import numpy as np
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

SIZE = 512
RESOLUTION = (SIZE, SIZE)
NUM_CLASSES = 3
BINARY_MASK = False
PIXEL_MEAN = [103.939, 116.779, 123.68]

def convert_image(image, mask = False, resnet50_preprocess = True):
  if mask:
    image = tf_image.decode_png(image, channels = 1)
    if BINARY_MASK:
      image = tf.where(tf.math.greater(image, 0), 1, 0)
    image.set_shape([None, None, 1])
    image = tf_image.resize(images = image, size = RESOLUTION)
  else:
    image = tf_image.decode_png(image, channels = 3)
    image.set_shape([None, None, 3])
    image = tf_image.resize(images = image, size = RESOLUTION)
    if resnet50_preprocess:
      # on reproduit le résultat de la fonction tf.keras.applications.resnet50.preprocess_input
      image = image[..., ::-1] - tf.constant(PIXEL_MEAN)
  return image

def predict_Deeplab(image, model, seuil):
  image = convert_image(image)
  prevision = tf.squeeze(model.predict(tf.expand_dims(image, 0), verbose=0))
  prev_mask = tf.where(tf.math.greater(tf.reduce_sum(prevision[:,:,1:], -1), seuil), 1, 0)
  return [prev_mask]