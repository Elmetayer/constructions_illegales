import tensorflow as tf

def predict_Unet(image, model, size_model, seuil):
  img_reduite = tf.image.resize(images = image, size = size_model)
  img_reduite /= 255

  prevision = tf.squeeze(model.predict(tf.expand_dims(img_reduite, 0), verbose=0))
  prev_mask = tf.where(tf.math.greater(tf.reduce_sum(prevision[:,:,1:], -1), seuil), 1, 0)
  return [prev_mask]