import numpy as np
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

SIZE = 512
RESOLUTION = (SIZE, SIZE)
BINARY_MASK = False
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]

def convert_image(image, mask = False, transformer_preprocess = True):
  '''
  fonction de conversion d'image pour SegFormer 
  '''
  if mask:
    image = tf_image.decode_png(image, channels = 1)
    image = tf_image.resize(images = image, size = RESOLUTION, method = 'nearest')
    if BINARY_MASK:
      image = tf.where(tf.math.greater(image, 0), 1, 0)
    image = tf.squeeze(image)
  else:
    image = tf_image.decode_png(image, channels = 3)
    image = tf_image.resize(images = image, size = RESOLUTION)
    image.set_shape([None, None, 3])
    if transformer_preprocess:
      # on reproduit le résultat de la fonction SegformerImageProcessor
      # https://github.com/huggingface/transformers/blob/f4f57f9dfa68948a383c352a900d588f63f6290a/src/transformers/models/segformer/image_processing_segformer.py
      # https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/image_transforms.py#L347
      # https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/utils/constants.py
      image /= 255
      image = (image - tf.constant(PIXEL_MEAN)) / tf.maximum(tf.constant(PIXEL_STD), backend.epsilon())
      image = tf.transpose(image, (2, 0, 1))
  return image

def predict_segformer(image, model, seuil):
  '''
  prévision Segformer
  '''
  img_reduite = convert_image(image)
  prevision_raw = np.transpose(np.squeeze(model.predict(tf.expand_dims(img_reduite, 0), verbose = 0).logits), (1, 2, 0))
  if seuil is None:
    prevision = tf.argmax(prevision_raw, axis=-1)
    prevision = tf.where(tf.math.greater(prevision, 0), 1, 0)
  else:
    prevision = tf.where(tf.math.greater(tf.reduce_sum(prevision_raw[:,:,1:], -1), seuil), 1, 0)
  return [prevision]