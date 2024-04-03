import numpy as np
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

RESOLUTION = (512, 512)

def predict_Unet(image, model, seuil):
  img_reduite = tf.image.resize(images = image, size = RESOLUTION)
  img_reduite /= 255

  prevision = tf.squeeze(model.predict(tf.expand_dims(img_reduite, 0), verbose=0))
  prev_mask = tf.where(tf.math.greater(tf.reduce_sum(prevision[:,:,1:], -1), seuil), 1, 0)
  return [prev_mask]