import numpy as np

SIZE_YOLO = 512

def predict_YOLOv8(image, model, seuil):
  '''
  on ne prédit que la classe "0", bâtiments
  '''
  res = model.predict(image, save = False, classes = [0], imgsz = SIZE_YOLO, conf = seuil, verbose=False)
  if len(res[0]) > 0:
    mask = np.sum(res[0].masks.data.cpu().numpy(), 0)
    mask = np.clip(mask, 0, 1)
  else:
    mask = [np.zeros((size_model, size_model))]
  return [mask]