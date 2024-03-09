def predict_YOLOv8(image, model, size_model = 512, seuil = 0.01):
  '''
  on ne prédit que la classe "0", bâtiments
  '''
  res = model.predict(image, save = False, classes = [0], imgsz = size_model, conf = seuil, verbose=False)
  return res