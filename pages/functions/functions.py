import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import geopandas as gpd
import shapely
import skimage as ski
import rasterio
from PIL import ImageOps
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon, label
import tensorflow as tf
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io

SIZE_YOLO = 512

def get_bbox(coords_center, size, mode):
  '''
  fonction qui calcule les coordonnées xmin, ymin, xmax, ymax de la bounding box
  à partir du point de référence, de la taille et du mode
  '''
  ccoords_center_WSG = gpd.GeoDataFrame(
      {'Nom': ['centre'],
      'geometry': [shapely.geometry.Point(coords_center[1], coords_center[0])]},
      crs = 'EPSG:4326')
  coords_center_Lambert = ccoords_center_WSG.to_crs('EPSG:2154')
  if mode == 'haut/gauche':
      bbox_Lambert = gpd.GeoDataFrame(
          {'Nom': ['min', 'max'],
          'geometry': [
              shapely.geometry.Point(coords_center_Lambert.geometry[0].x, coords_center_Lambert.geometry[0].y - size),
              shapely.geometry.Point(coords_center_Lambert.geometry[0].x + size, coords_center_Lambert.geometry[0].y)]},
          crs = 'EPSG:2154')
  if mode == 'centre':
      bbox_Lambert = gpd.GeoDataFrame(
          {'Nom': ['min', 'max'],
          'geometry': [
              shapely.geometry.Point(coords_center_Lambert.geometry[0].x - size//2, coords_center_Lambert.geometry[0].y - size//2),
              shapely.geometry.Point(coords_center_Lambert.geometry[0].x + size//2, coords_center_Lambert.geometry[0].y + size//2)]},
          crs = 'EPSG:2154')
  bbox_WSG = bbox_Lambert.to_crs('EPSG:4326')
  return(bbox_WSG.geometry[0].x, bbox_WSG.geometry[0].y, bbox_WSG.geometry[1].x, bbox_WSG.geometry[1].y)

def get_bbox_center(bbox):
  '''
  renvoie le centre de la bounding box
  '''
  return([(bbox[1] + bbox[3])/2, (bbox[0] + bbox[2])/2])

def get_bbox_Lambert(bbox):
  '''
  fonction qui renvoie un gdf en Lambert à partir d'une bbox en WSG84
  '''
  coords_bbox_WSG = gpd.GeoDataFrame({
    'Nom': ['min', 'max'],
    'geometry': [
        shapely.geometry.Point(bbox[0], bbox[1]),
        shapely.geometry.Point(bbox[2], bbox[3])]},
    crs = 'EPSG:4326')
  coords_bbox_Lambert = coords_bbox_WSG.to_crs('EPSG:2154')
  return(coords_bbox_Lambert.geometry[0].x, coords_bbox_Lambert.geometry[1].x, coords_bbox_Lambert.geometry[0].y, coords_bbox_Lambert.geometry[1].y)

def calcul_ious_shapes(shapes_1_ext, shapes_2_ext, seuil_iou = 0):
  '''
  fonction qui calcule l'IoU des shapes_1_ext en cherchant à rapprocher les shapes_2_ext
  renvoie la liste des ious calculés
  '''
  ious = []
  rapprochements = []
  selected_intersections = []
  for i in range(len(shapes_1_ext)):
    shape_1 = shapely.Polygon(shapes_1_ext[i])
    unions = []
    unions.append(shape_1)
    intersections = []
    for j in range(len(shapes_2_ext)):
      shape_2 = shapely.Polygon(shapes_2_ext[j])
      if shape_2 is not None:
        if shape_1.intersects(shape_2):
          intersection = shape_1.intersection(shape_2)
          intersections.append(intersection)
          unions.append(shape_2)
    if len(intersections) > 0:
      all_intersections = gpd.GeoSeries(intersections, crs = 2154).unary_union
      intersection_area = shapely.area(all_intersections)
    else:
      intersection_area = 0
    union_area = shapely.area(gpd.GeoSeries(unions, crs = 2154).unary_union)
    if union_area == 0:
      ious.append(0)
    else:
      iou = intersection_area/union_area
      ious.append(iou)
      if len(intersections) > 0 and iou > seuil_iou:
        selected_intersections.append(all_intersections)
    rapprochements.append(len(intersections))
  return ious, rapprochements, selected_intersections

def isInMap(xrange, yrange, bounds = False):
  '''
  fonction qui renvoie une autre fonction permettant de tester si une forme
  est à l'intérieur d'un restangle défini par xrange, yrange
  si l'argument "bounds" est mis à True, la fonction renvoyée réalise le test sur la base de l'emprise rectangulaire de la forme
  sinon, la fonction renvoyée réalise le test sur la base du centroïd
  l'argument "bounds" est à False par défaut
  '''
  def my_function(polynom):
      if bounds:
          minx, miny, maxx, maxy = polynom.bounds
      else:
          centroid_x, centroid_y = polynom.centroid.x, polynom.centroid.y
          minx, miny, maxx, maxy = centroid_x, centroid_y, centroid_x, centroid_y
      if xrange[0] < minx and xrange[1] > maxx and yrange[0] < miny and yrange[1] > maxy:
          return(True)
      else:
          return(False)
  return my_function

def affiche_contours(
    image, predict_function, model, size_model, coord_transform, gdf_shapes_ref, resolution_target = (1000, 1000),
    seuil = 0.05, seuil_iou = 0.01, 
    seuil_area = 10,
    tolerance_polygone = 0.1, tolerance_display = 1):
  '''
  fonction qui calcule les IoU des formes prédites par rapport à des formes de référence
  utilise les mêmes arguements que compare_ligne, avec en plus :
  - predict_function : fonction qui prend en argument le triplet (image, model, seuil) et qui renvoie une liste de prévisions du modèle "model" sous forme
  de masque numpy.array avec les dimensions correspondant au paramètre "resolution_model", pour le seuil de détection "seuil"
  /!\ predict_function doit pouvoir faire l'inférence sur l'image et non pas sur le chemin file_name
  - image : array de l'image sur laquelle faire la prédiction
  - seuil_iou : les formes avec un IoU inférieur sont affichées en rouge
  - delta_only : si True, n'affiche que les formes en rouge
  - coord_transform : transformation affine pour passer des pixels aux coordonnées géographiques cible
  - gdf_shapes_ref : fichier de shapefiles à utiliser pour la comparaison des formes prédites > par défaut, on compare par rapport aux formes du df_decoupe
  renvoie :
  - geoSeries avec les formes prédites, crs 2154, avec uniquement les contours
  - geoSeries avec les formes de réference, crs 2154, avec uniquement les contours, et en fusionnant les formes adjacentes
  - shapes_pred_ious
  - shapes_ref_ious
  - le calcul de l'iou "pixel par pixel"
  '''

  # Prévision
  prev_masks = predict_function(image, model, size_model, seuil)
  
  # Contours prédiction
  mask_contours = []
  for prev_mask in prev_masks:
    prev_mask_resized = tf.squeeze(tf_image.resize(images = np.expand_dims(prev_mask, -1), size = resolution_target, method = 'nearest'))
    mask_padded = np.pad(prev_mask_resized, ((1, 1),(1, 1)), mode = 'constant', constant_values = 0)
    mask_contours += ski.measure.find_contours(image = mask_padded == 1)

  # utilisation des informations raster pour les coordonnées
  X0, Y0, coords_scale = coord_transform
  raster_transform = rasterio.transform.Affine(coords_scale, 0.0, X0,
                            0.0, -coords_scale, Y0 + coords_scale*resolution_target[1])
  bounds = rasterio.coords.BoundingBox(X0, Y0, X0 + coords_scale*resolution_target[0], Y0 + coords_scale*resolution_target[1])

  # Shapes référence
  # on enlève les shapes extérieurs à la dalle pour diminuer le volume de données inutiles
  '''
  img_bound = shapely.Polygon(((bounds.left, bounds.bottom), (bounds.right, bounds.bottom), (bounds.right, bounds.top), (bounds.left, bounds.top), (bounds.left, bounds.bottom)))
  try:
    gdf_shapes_ref['geometry'] = gdf_shapes_ref['geometry'].intersection(img_bound)
    gdf_shapes_ref = gdf_shapes_ref[
      ~(gdf_shapes_ref['geometry'].isna() | gdf_shapes_ref['geometry'].is_empty)]
  except:
    # si erreur, on fait un test simple
    gdf_shapes_ref = gdf_shapes_ref[gdf_shapes_ref['geometry'].apply(isInMap([bounds.left, bounds.right], [bounds.bottom, bounds.top], False))]
  '''

  # Shapes prédiction
  raster_transformer = rasterio.transform.AffineTransformer(raster_transform)
  shapes_xy = []
  shapes_predict = []
  # parcours de tous les contours prédits
  for contour in mask_contours:
    # on crée le polygone
    polygon = approximate_polygon(contour, tolerance = tolerance_polygone)
    # on tranforme en coordonnées
    xy_polygon = raster_transformer.xy(polygon[:,0], polygon[:,1])
    shapes_xy.append(xy_polygon)
    # on crée le polygone
    shapes_predict.append(shapely.Polygon(np.array(xy_polygon).transpose()))
  # on filtre les surface trop petites
  shapes_predict = [shape for shape in shapes_predict if shapely.area(shape) > seuil_area]
  # Ajout des trous dans les shapes prédiction
  shapes_predict_holes = []
  shapes_holes = []
  for shape_a in shapes_predict:
    if shape_a not in shapes_holes:
      shape_a_holes = []
      for shape_b in shapes_predict:
        if shape_a.contains_properly(shape_b):
          shape_a_holes.append(shape_b.exterior)
          shapes_holes.append(shape_b)
      if len(shape_a_holes) > 0:
        shapes_predict_holes.append(shapely.Polygon(shape_a.exterior, holes = shape_a_holes))
      else:
        shapes_predict_holes.append(shape_a)
  gdf_shapes_predict = gpd.GeoDataFrame(geometry = gpd.GeoSeries(shapes_predict_holes, crs=2154), crs=2154)

  # Intersection et IoUs
  shapes_ref = gdf_shapes_ref['geometry'].exterior
  shapes_ref = [shape for shape in shapes_ref if shape is not None]
  shapes_predict = gdf_shapes_predict['geometry'].exterior
  # pour les formes prédites, on simplifie
  shapes_predict = [shape.simplify(tolerance_display) for shape in shapes_predict if shape is not None]
  # iou des prédictions
  shapes_pred_ious, shapes_pred_rapprochements, _ = calcul_ious_shapes(shapes_predict, shapes_ref)
  # iou des réferences
  shapes_ref_ious, shapes_ref_rapprochements, _ = calcul_ious_shapes(shapes_ref, shapes_predict)

  # génération du graphique
  nb_formes = len(shapes_ref)
  fig = px.imshow(
      ImageOps.flip(image),
      x = np.linspace(bounds.left, bounds.right, resolution_target[0]),
      y = np.linspace(bounds.bottom, bounds.top, resolution_target[1]),
      title = '{} bâtiments référence<br>{} zones détectées'.format(
          str(nb_formes),
          np.sum((np.array(shapes_pred_ious) <= seuil_iou) & (np.array(shapes_pred_rapprochements) == 0))),
      origin = 'lower')
  # ajout des formes
  shape_traces_to_plot = []
 
  # formes prédites
  i_pred = 0
  i_pred_delta = 0
  for shape, iou, rapprochement in zip(shapes_predict, shapes_pred_ious, shapes_pred_rapprochements):
    x_coords, y_coords = shape.xy
    if iou <= seuil_iou:
      shape_traces_to_plot.append(
        go.Scatter(
            x = x_coords.tolist(),
            y = y_coords.tolist(),
            line = dict(color='black', width=1),
            mode = 'lines',
            fill = 'toself',
            fillcolor = 'red',
            opacity = 0.4,
            text = 'iou prédiction: {}'.format(iou),
            hoverinfo = 'text',
            name = 'écart',
            legendgroup = 'écart',
            showlegend = (i_pred_delta==0)))
      i_pred_delta += 1
    else:
      shape_traces_to_plot.append(
        go.Scatter(
            x = x_coords.tolist(),
            y = y_coords.tolist(),
            line = dict(color='black', width=1),
            mode = 'lines',
            fill = 'toself',
            fillcolor = '#ffed6f',
            opacity = 0.4,
            text = 'iou prédiction: {}<br>{} bâtiments rapprochés'.format(iou, rapprochement),
            hoverinfo = 'text',
            name = 'prédiction',
            legendgroup = 'prédiction',
            showlegend = (i_pred==0)))
      i_pred += 1

  # formes de référence
  #for i, (shape, iou, rapprochement) in enumerate(zip(shapes_predict, shapes_pred_ious, shapes_pred_rapprochements)):
  for i, (shape, iou, rapprochement) in enumerate(zip(shapes_ref, shapes_ref_ious, shapes_ref_rapprochements)):
    x_coords, y_coords = shape.xy
    shape_traces_to_plot.append(
      go.Scatter(
        x = x_coords.tolist(),
        y = y_coords.tolist(),
        name = 'référence',
        legendgroup = 'référence',
        showlegend = (i==0)))
    '''
        line = dict(color='black', width=1),
        mode = 'lines',
        fill = 'toself',
        fillcolor = 'blue',
        opacity = 0.4,
        text = 'iou référence: {}<br>{} prédictions rapprochées'.format(iou, rapprochement),
        hoverinfo = 'text',
    '''

  fig.add_traces(shape_traces_to_plot)
  # mise en forme
  fig.update_layout(
    xaxis=dict(title='X en Lambert93'),
    yaxis=dict(title='Y en Lambert93'),
    plot_bgcolor='white',
    height = 600,
    width = 600)

  return shapes_predict, shapes_ref, shapes_pred_ious, shapes_ref_ious, shapes_pred_rapprochements, shapes_ref_rapprochements, fig

def predict_YOLOv8(image, model, size_model = SIZE_YOLO, seuil = 0.01):
  '''
  on ne prédit que la classe "0", bâtiments
  '''
  res = model.predict(image, save = False, classes = [0], imgsz = size_model, conf = seuil, verbose=False)
  if len(res[0]) > 0:
    mask = np.sum(res[0].masks.data.cpu().numpy(), 0)
    mask = np.clip(mask, 0, 1)
  else:
    mask = [np.zeros((size_model, size_model))]
  return [mask]