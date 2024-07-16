import numpy as np
import pandas as pd
import cv2
import matplotlib as mpl

import torch
import torchvision

import ultralytics 
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression

from pages.functions import config

class yolov8_ActivationsAndGradients:

    def __init__(self, model, target_layers, conf_threshold, n_classes, predict_classes):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # voir https://github.com/pytorch/pytorch/issues/61519 : il n'est pas possible d'utiliser un backward hook
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))
        self.conf_threshold = conf_threshold
        self.n_classes = n_classes
        self.predict_classes = predict_classes

    def save_activation(self, module, input, output):
      activation = output
      # /!\ on peut ajouter les résultats à la suite car on est en "forward"
      self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
      if not hasattr(output, 'requires_grad') or not output.requires_grad:
        return

      def _store_grad(grad):
        # /!\ il faut ajouter les résultats dans le sens inverse car on est en "backward"
        self.gradients = [grad.cpu().detach()] + self.gradients

      output.register_hook(_store_grad)

    def post_process(self, result):
      # batch = 1
      x = result[0][0]
      x = x.transpose(-1, -2)
      # on filtre sur le seuil de confiance et la classe cible
      confs, j = x[:,4:4 + self.n_classes].max(1, keepdim=True)
      x = x[(torch.logical_and(j == torch.tensor([0], device=x.device), confs > self.conf_threshold)).any(1)]
      # séparation des données
      box, cls, logits = x.split((4, self.n_classes, x.shape[1] - self.n_classes - 4), 1)
      # autre donnée
      result_data = result[1][1][0]
      result_data = result_data.transpose(-1, -2)
      # on filtre sur le seuil de confiance et la classe cible
      result_data = result_data[(torch.logical_and(j == torch.tensor([0], device=x.device), confs > self.conf_threshold)).any(1)]
      # proto /!\ pas de gradient possible
      # proto_data = result[1][-1][0]
      # proto_data = torch.reshape(proto_data, (proto_data.shape[0], -1))
      return box, cls, logits, result_data

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        boxes, conf, logits, result_data = self.post_process(model_output)
        return [boxes, conf, logits, result_data]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolov8_GradCamLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, result_data):
      loss = 0
      for data in result_data:
        for i in range(data.shape[0]):
          for j in range(data.shape[1]):
            loss += data[i, j]
      return loss

class yolov8_GradCamLoss_box(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, result_data, boxes_data):
      loss = 0
      for data in result_data:
        for i in range(boxes_data.shape[0]):
          x1 = boxes_data[i][0]
          y1 = boxes_data[i][1]
          x2 = boxes_data[i][2]
          y2 = boxes_data[i][3]
          loss += data[y1:y2, x1:x2].sum()
      return loss

def get_last_conv_layers_YOLO(model_YOLO):
  all_last_layers = model_YOLO.model[22].cv2 + model_YOLO.model[22].cv3 + model_YOLO.model[22].cv4
  all_last_layers_blocks = ['cv2']*len(model_YOLO.model[22].cv2) + ['cv3']*len(model_YOLO.model[22].cv3) + ['cv4']*len(model_YOLO.model[22].cv4)
  target_layers_info = [(layer, block_name + '_' + str(j) + '_' + str(i)) for (j, (sequence, block_name)) in enumerate(zip(all_last_layers, all_last_layers_blocks)) for i, layer in enumerate(sequence) if type(layer) in [ultralytics.nn.modules.conv.Conv, torch.nn.modules.conv.Conv2d]]
  target_layers = [layer for (layer, _) in target_layers_info]

  return target_layers


def make_gradCam_heatmap_YOLO(image, model_GradCam, model, target_layers, conf_threshold, result_display, 
                         normalize_boxes = False, abs_norm = False, norm_grads_act = False,
                         boxes_only = False, abs_values = False,
                         predict_classes = config.model_YOLO.YOLO_PREDICT_CLASSES,
                         names_result = config.gradcam_YOLO.OUTPUT_YOLO,
                         img_weigth = config.gradcam_YOLO.IMG_WEIGHT):
  # préparation des images
  image_array = np.asarray(image)
  # on redimensionne l'image de départ à la taille de la sortie du modèle
  image_array_resized = cv2.resize(image_array, config.model_YOLO.YOLO_RESOLUTION)
  # pour les calculs de gradient
  input_tensor = torch.from_numpy(np.expand_dims(np.asarray(torchvision.transforms.functional.resize(image, config.model_YOLO.YOLO_RESOLUTION)).transpose(2, 0, 1)/255, 0)).to(torch.float32)

  # coordonnées des boxes prédites par le modèle
  result = model.predict(input_tensor, save = False, classes = predict_classes, imgsz = config.model_YOLO.YOLO_SIZE, conf = conf_threshold, iou = config.gradcam_YOLO.IOU_THRESHOLD_YOLO, verbose=False)
  boxes_coords = result[0].boxes.data[:,:4].cpu().detach().numpy().astype(np.int32)

  # données de sorties de la fonction
  dict_heatmaps = {}

  # initialisation du modèle pour le calcul de gradient
  for weight in model_GradCam.parameters():
    weight.requires_grad = True
  model_GradCam.eval()

  # passe forward
  n_classes = len(model.names)
  activations_and_grads = yolov8_ActivationsAndGradients(model_GradCam, target_layers, conf_threshold, n_classes, predict_classes)
  forward_results = activations_and_grads(input_tensor)

  # on ne poursuit que s'il y a des résultats
  if forward_results[0].shape[0] > 0:
    forward_results = forward_results + [forward_results]

    # passe backward pour toutes les combinaisons layer x type de résultat
    for forward_result, name_result in zip(forward_results, names_result):
      # dictionnaire des valeurs
      dict_heatmaps[name_result] = {}
      dict_heatmaps[name_result]['layers'] = {}

      # échelle pour la normalisation des valeurs
      heatmap_min = 0
      heatmap_max = 0

      # passe backward
      if boxes_only:
        loss_function = yolov8_GradCamLoss_box()
        model_GradCam.zero_grad()
        if isinstance(forward_result, list):
          loss = loss_function(forward_result, boxes_coords)
        else:
          loss = loss_function([forward_result], boxes_coords)
      else:
        loss_function = yolov8_GradCamLoss()
        model_GradCam.zero_grad()
        if isinstance(forward_result, list):
          loss = loss_function(forward_result)
        else:
          loss = loss_function([forward_result])
      loss.backward(retain_graph=True)

      # récupération des activations et des gradients pour chaque target layer
      activations_list = [a.cpu().data.numpy() for a in activations_and_grads.activations]
      grads_list = [g.cpu().data.numpy() for g in activations_and_grads.gradients]
      for id_layer, (activations, grads) in enumerate(zip(activations_list, grads_list)):
        dict_heatmaps[name_result]['layers'][id_layer] = {}
        if norm_grads_act:
          dict_heatmaps[name_result]['layers'][id_layer]['activations'] = np.squeeze(activations)
          dict_heatmaps[name_result]['layers'][id_layer]['gradients'] = np.squeeze(grads)
        else:
          dict_heatmaps[name_result]['layers'][id_layer]['activations'] = (np.squeeze(activations) - np.min(activations)) / (np.max(activations) - np.min(activations))
          dict_heatmaps[name_result]['layers'][id_layer]['gradients'] = (np.squeeze(grads) - np.min(grads)) / (np.max(grads) - np.min(grads))
        dict_heatmaps[name_result]['layers'][id_layer]['cam_heatmap'] = np.squeeze(activations) * np.squeeze(grads)
        # mise à jour du min/max
        heatmap_min = min(heatmap_min, np.min(dict_heatmaps[name_result]['layers'][id_layer]['cam_heatmap']))
        heatmap_max = max(heatmap_max, np.max(dict_heatmaps[name_result]['layers'][id_layer]['cam_heatmap']))

      # sauvegarde du min/max
      dict_heatmaps[name_result]['heatmap_min'] = heatmap_min
      dict_heatmaps[name_result]['heatmap_max'] = heatmap_max

      # réinitialisation des gradients
      activations_and_grads.gradients = []

    # création des images superposées
    all_normalized_heatmap = []
    for id_layer in range(len(target_layers)):
      for name_result in names_result:
        if id_layer in dict_heatmaps[name_result]['layers'].keys():
          heatmap_data = np.sum(dict_heatmaps[name_result]['layers'][id_layer][result_display], 0)
          heatmap = cv2.resize(heatmap_data, config.model_YOLO.YOLO_RESOLUTION)
          if normalize_boxes:
            # on extrait la heatmap pour chaque box
            blank_heatmaps = []
            for i in range(boxes_coords.shape[0]):
              blank_heatmap = np.zeros(RESOLUTION, dtype=np.float32)
              x1 = boxes_coords[i][0]
              y1 = boxes_coords[i][1]
              x2 = boxes_coords[i][2]
              y2 = boxes_coords[i][3]
              heatmap_crop = heatmap[y1:y2, x1:x2].copy()
              blank_heatmap[y1:y2, x1:x2] = heatmap_crop
              blank_heatmaps.append([blank_heatmap])
            heatmap = np.max(np.concatenate(blank_heatmaps), 0)
          if abs_norm:
            normalized_heatmap = (heatmap - dict_heatmaps[name_result]['heatmap_min']) / (dict_heatmaps[name_result]['heatmap_max'] - dict_heatmaps[name_result]['heatmap_min'])
          else:
            normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
          all_normalized_heatmap.append(normalized_heatmap)
          # conversion de la heatmap en couleurs
          jet = mpl.colormaps['jet']
          jet_colors = jet(np.arange(256))[:, :3]
          jet_heatmap = jet_colors[np.uint8(255 * normalized_heatmap)] * 255
          superposed_img = jet_heatmap * (1 - img_weigth) + image_array_resized * img_weigth
          dict_heatmaps[name_result]['layers'][id_layer]['superposed_heatmap'] = np.uint8(superposed_img)
        else:
          dict_heatmaps[name_result]['layers'][id_layer] = {}
          dict_heatmaps[name_result]['layers'][id_layer]['superposed_heatmap'] = np.zeros(config.model_YOLO.YOLO_RESOLUTION + (3,), dtype=np.uint8)

      # création de la heatmap moyenne
      normalized_heatmaps = np.concatenate([np.expand_dims(normalized_heatmap, 0) for normalized_heatmap in all_normalized_heatmap])
      normalized_heatmap_mean = np.mean(normalized_heatmaps, 0)
      jet = mpl.colormaps['jet']
      jet_colors = jet(np.arange(256))[:, :3]
      jet_heatmap = jet_colors[np.uint8(255 * normalized_heatmap_mean)] * 255
      superposed_img = jet_heatmap * (1 - img_weigth) + image_array_resized * img_weigth
      dict_heatmaps[name_result]['mean_superposed_heatmap'] = np.uint8(superposed_img)

    return dict_heatmaps
