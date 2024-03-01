def predict_YOLOv8_flat(file_name, model, seuil):
  '''
  on ne prédit que la classe "0", bâtiments
  seul un masque "applati" est retourné, mais dans une liste (de longueur 1)
  '''
  res = model.predict(file_name, save = False, classes = [0], imgsz = SIZE, conf = seuil, verbose=False)
  if len(res[0]) > 0:
    mask_res = tf.convert_to_tensor(torch.sum(res[0].cuda().masks.data, 0).cpu())
    mask_list = [tf.clip_by_value(mask_res, 0, 1)]
  else:
    mask_list = [tf.zeros((SIZE, SIZE))]
  return mask_list

def affiche_contours_iou(
    model, predict_function, df, chemin_images, index_l, resolution_model, resolution_target = (1000, 1000),
    image = None,
    seuil = None, seuil_iou = 0, coord_transform = None, gdf_shapes_ref = None, affichage = True, delta_only = False,
    seuil_area = 10,
    types_ref = ['BATIMENT'],
    tolerance_polygone = 0.1):
  '''
  fonction qui calcule les IoU des formes prédites par rapport à des formes de référence
  utilise les mêmes arguements que compare_ligne, avec en plus :
  - predict_function : fonction qui prend en argument le triplet (image, model, seuil) et qui renvoie une liste de prévisions du modèle "model" sous forme
  de masque numpy.array avec les dimensions correspondant au paramètre "resolution_model", pour le seuil de détection "seuil"
  /!\ predict_function doit pouvoir faire l'inférence sur l'image et non pas sur le chemin file_name
  - image : array de l'image sur laquelle faire la prédiction, si renseigné, utilisé à la place de l'image à l'index index_l
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
  # image sur laquelle faire l'inférence
  if image is None:
    file_name = chemin_images + df.loc[index_l,'fichier_img']
    image = Image.open(file_name)

  # Prévision
  prev_masks = predict_function(image, model, seuil)
  # Contours prédiction
  mask_contours = []
  for prev_mask in prev_masks:
    prev_mask_resized = tf.squeeze(tf_image.resize(images = np.expand_dims(prev_mask, -1), size = resolution_target, method = 'nearest'))
    mask_padded = np.pad(prev_mask_resized, ((1, 1),(1, 1)), mode = 'constant', constant_values = 0)
    mask_contours += ski.measure.find_contours(image = mask_padded == 1)

  # utilisation des informations raster pour les coordonnées
  if coord_transform is None:
    raster_name = chemin_images + df.loc[index_l, 'fichier_raster']
    with rasterio.open(raster_name) as raster:
      # création du convertisseur pour passer des coordonnées en pixels
      raster_transform = raster.transform
      bounds = raster.bounds
  else:
    X0, Y0, coords_scale = coord_transform
    raster_transform = rasterio.transform.Affine(coords_scale, 0.0, X0,
                              0.0, -coords_scale, Y0 + coords_scale*resolution_target[1])
    bounds = rasterio.coords.BoundingBox(X0, Y0, X0 + coords_scale*resolution_target[0], Y0 + coords_scale*resolution_target[1])

  # Shapes référence
  if gdf_shapes_ref is None:
    shape_name = chemin_images + df.loc[index_l, 'fichier_shapes']
    gdf_shapes_ref_copy = gpd.read_file(shape_name)
  else:
    gdf_shapes_ref_copy = gdf_shapes_ref.copy()
  # on ne garde que le type de formes de référence spécifié, si applicable
  if types_ref is not None:
    gdf_shapes_ref_copy = gdf_shapes_ref_copy[gdf_shapes_ref_copy['Type'].isin(types_ref)]
  # on s'assure que la colonne 'Type' est présente pour l'affichage
  if 'Type' not in gdf_shapes_ref_copy.columns:
    gdf_shapes_ref_copy['Type'] = 'inconnu'
  # on enlève les shapes extérieurs à la dalle pour diminuer le volume de données inutiles
  img_bound = shapely.Polygon(((bounds.left, bounds.bottom), (bounds.right, bounds.bottom), (bounds.right, bounds.top), (bounds.left, bounds.top), (bounds.left, bounds.bottom)))
  try:
    gdf_shapes_ref_copy['geometry'] = gdf_shapes_ref_copy['geometry'].intersection(img_bound)
    gdf_shapes_ref_copy = gdf_shapes_ref_copy[
      ~(gdf_shapes_ref_copy['geometry'].isna() | gdf_shapes_ref_copy['geometry'].is_empty)]
  except:
    # si erreur, on fait un test simple
    gdf_shapes_ref_copy = gdf_shapes_ref_copy[gdf_shapes_ref_copy['geometry'].apply(isInMap([bounds.left, bounds.right], [bounds.bottom, bounds.top], False))]

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

  # /!\ on court-circuite le traitement de fusion pour éviter de supprimer des formes de référence
  '''
  # Regroupement et fusion des formes adjacentes
  gdf_shapes_ref['group'] = ''
  for index in range(gdf_shapes_ref.shape[0]):
    group = gdf_shapes_ref[~gdf_shapes_ref.geometry.disjoint(gdf_shapes_ref.geometry[index])].index.to_list()
    if len(group) > 1:
      if gdf_shapes_ref.loc[index, 'group'] == '':
        gdf_shapes_ref.loc[group, 'group'] = 'group_' + str(index)
      else:
        gdf_shapes_ref.loc[group, 'group'] = gdf_shapes_ref.loc[index, 'group']
  gdf_shapes_ref_merged = gdf_shapes_ref.dissolve(by = 'group', as_index = False)
  gdf_shapes_ref_merged = gdf_shapes_ref_merged[gdf_shapes_ref_merged['geometry'].is_valid].reset_index()
  '''

  # Intersection et IoUs
  shapes_ref = gdf_shapes_ref_copy['geometry'].exterior
  shapes_ref = [shape for shape in shapes_ref if shape is not None]
  shapes_predict = gdf_shapes_predict['geometry'].exterior
  shapes_predict = [shape for shape in shapes_predict if shape is not None]
  # iou des prédictions
  shapes_pred_ious, shapes_pred_rapprochements, _ = calcul_ious_shapes(shapes_predict, shapes_ref)
  # iou des réferences
  shapes_ref_ious, shapes_ref_rapprochements, _ = calcul_ious_shapes(shapes_ref, shapes_predict)

  # IoU "pixel par pixel"
  # Masque
  mask_name = chemin_images + df.loc[index_l,'fichier_mask']
  image_mask = tf.io.read_file(mask_name)
  image_mask = tf.image.decode_png(image_mask, channels = 1)
  image_mask = tf.where(tf.math.greater(image_mask, 0), 1, 0)
  mask_reduit = tf.squeeze(tf.image.resize(images = image_mask, size = resolution_model, method = 'nearest'))

  # Métrique IoU
  prev_mask = prev_masks[0]
  m_IoU = calcul_iou_mask(mask_reduit, prev_mask)

  fig = None
  if affichage:
    # génération du graphique
    X0 = bounds.left
    Y0 = bounds.top
    nb_formes = len(shapes_ref)
    fig = px.imshow(
        #cv2.flip(raster_data, 0),
        ImageOps.flip(image),
        x = np.linspace(bounds.left, bounds.right, resolution_target[0]),
        y = np.linspace(bounds.bottom, bounds.top, resolution_target[1]),
        title = 'Image {}, {}<br>{} bâtiments référence<br>{} zones détectées'.format(
            str(X0), str(Y0), str(nb_formes),
            np.sum((np.array(shapes_pred_ious) <= seuil_iou) & (np.array(shapes_pred_rapprochements) == 0))),
        origin = 'lower')
    # ajout des formes
    shape_traces_to_plot = []
    if not delta_only:
      # formes de référence
      for i, (shape, iou, rapprochement, type_shape) in enumerate(zip(shapes_ref, shapes_ref_ious,
                                                          shapes_ref_rapprochements,
                                                          gdf_shapes_ref_copy['Type'])):
        list_x, list_y = shape.xy
        shape_traces_to_plot.append(
            go.Scatter(
                x = np.array(list_x),
                y = np.array(list_y),
                line = dict(color='black', width=1),
                mode = 'lines',
                fill = 'toself',
                fillcolor = '#80b1d3',
                opacity = 0.4,
                text = 'type: {}<br>iou référence: {}<br>{} prédictions rapprochées'.format(type_shape, iou, rapprochement),
                hoverinfo = 'text',
                name = 'référence',
                legendgroup = 'référence',
                showlegend = (i==0)))
    # formes prédites
    i_pred = 0
    i_pred_delta = 0
    for i, (shape, iou, rapprochement) in enumerate(zip(shapes_predict, shapes_pred_ious, shapes_pred_rapprochements)):
      list_x, list_y = shape.xy
      if iou <= seuil_iou:
        shape_traces_to_plot.append(
          go.Scatter(
              x = np.array(list_x),
              y = np.array(list_y),
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
      elif not delta_only:
        shape_traces_to_plot.append(
          go.Scatter(
              x = np.array(list_x),
              y = np.array(list_y),
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

    fig.add_traces(shape_traces_to_plot)
    # mise en forme
    fig.update_layout(
        xaxis=dict(title='X en Lambert93'),
        yaxis=dict(title='Y en Lambert93'),
        plot_bgcolor='white',
        height = 900,
        width = 900)

  return shapes_predict, shapes_ref, shapes_pred_ious, shapes_ref_ious, shapes_pred_rapprochements, shapes_ref_rapprochements, m_IoU, fig

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

def calcul_iou_mask(masque1, masque2, min_pixel = 1):
  if not masque1.shape==masque2.shape:
    print("masques de tailles incompatibles")
    return None
  else:
    M1 = masque1 > 0
    M2 = masque2 > 0
    return (min_pixel + np.sum(M1 & M2))/(min_pixel + np.sum(M1 | M2))
  return ious, rapprochements, selected_intersections