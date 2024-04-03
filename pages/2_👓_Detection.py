import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
import requests
from io import BytesIO
import geopandas as gpd
from PIL import Image
from ultralytics import YOLO

from pages.functions.functions import *

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

@st.cache_data(show_spinner = False)
def get_IGN_data(xmin, xmax, ymin, ymax, pixel_size):
   if all((xmin, xmax, ymin, ymax, pixel_size)):
      request_wms = 'https://data.geopf.fr/wms-r?LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}'.format(
      xmin, ymin, xmax, ymax, pixel_size, pixel_size)
      response_wms = requests.get(request_wms).content
      orthophoto = Image.open(BytesIO(response_wms))
      bounds = gpd.GeoDataFrame(
         {'Nom': ['name1', 'name2'],
         'geometry': [shapely.geometry.Point(xmin, ymin), shapely.geometry.Point(xmax, ymax)]},
         crs = 'EPSG:2154')
      bounds = bounds.to_crs('EPSG:4326')
      request_wfs = 'https://data.geopf.fr/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&typename=CADASTRALPARCELS.PARCELLAIRE_EXPRESS:batiment&outputformat=application/json&BBOX={},{},{},{}'.format(
    bounds.geometry[0].y, bounds.geometry[0].x, bounds.geometry[1].y, bounds.geometry[1].x)
      response_wfs = requests.get(request_wfs)
      cadastre = response_wfs.json()
      return orthophoto, cadastre
   else:
      return None, None

@st.cache_data(show_spinner = False)
def get_fig_prev(xmin, ymin, pixel_size, scale, _orthophoto, cadastre):
   if (xmin, ymin, pixel_size, scale, _orthophoto, cadastre) != (None, None, None, None, None, None):
      gdf_cadastre = gpd.GeoDataFrame.from_features(cadastre['features'])
      if gdf_cadastre.shape[0]>0 :
         gdf_cadastre = gdf_cadastre.set_crs('EPSG:4326').to_crs('EPSG:2154')
         gdf_cadastre['geometry'] = gdf_cadastre['geometry'].make_valid()
         gdf_cadastre = gdf_cadastre.explode(index_parts = False)
         gdf_cadastre = gdf_cadastre[gdf_cadastre['geometry'].geom_type.isin(['Polygon', 'MultiPolygon'])]
      _, _, _, _, _, _, fig = affiche_contours(
         _orthophoto, predict_YOLOv8, model_YOLO, SIZE_YOLO, 
         (xmin, ymin, scale), gdf_shapes_ref = gdf_cadastre,
         resolution_target = (pixel_size, pixel_size),
         seuil = 0.05, seuil_iou = 0.01,
         seuil_area = 10,
         tolerance_polygone = 0.1)
      return fig
   else:
      return None

# titre de la page
st.set_page_config(page_title="Détection", page_icon="👓", layout = 'wide')

# variables de session
PIXEL_SIZE_MIN = 500
PIXEL_SIZE_MAX = 2000
PIXEL_SIZE_DEFAULT = 1000
PIXEL_SCALE_REF = 0.2
SIZE_MAX = 1000

if 'bbox_selected' not in st.session_state:
   st.session_state['bbox_selected'] = None
if 'coords_bbox_Lambert' not in st.session_state:
   if st.session_state['bbox_selected'] is None:
      st.session_state['coords_bbox_Lambert'] = (None, None, None, None)
   else:
      st.session_state['coords_bbox_Lambert'] = get_bbox_Lambert(st.session_state['bbox_selected'])
if 'pixel_size' not in st.session_state:
   st.session_state['pixel_size'] = PIXEL_SIZE_DEFAULT
if 'scale' not in st.session_state:
   if all(st.session_state['coords_bbox_Lambert']):
      st.session_state['scale'] = (st.session_state['coords_bbox_Lambert'][1] - st.session_state['coords_bbox_Lambert'][0])/st.session_state['pixel_size']
   else:
      st.session_state['scale'] = None
if 'fig' not in st.session_state:
   st.session_state['fig'] = None
if 'cadastre' not in st.session_state:
   st.session_state['cadastre'] = None
if 'orthophoto' not in st.session_state:
   st.session_state['orthophoto'] = None


inter1, col1, inter2, col2, inter3 = st.sidebar.columns([1, 6, 1, 4, 1])
with col1:
  load_button = st.button('données IGN')
with col2:
  calcul_button = st.button('prédire')

#################
# données IGN #
#################

# mise à jour des données IGN
if load_button:
   if 'bbox' in st.session_state:
      st.session_state['bbox_selected'] = st.session_state['bbox']
   if all((st.session_state['bbox_selected'], st.session_state['pixel_size'])):
      st.session_state['coords_bbox_Lambert'] = get_bbox_Lambert(st.session_state['bbox_selected'])
      with st.spinner('récupération des données IGN ...'):
         st.session_state['orthophoto'], st.session_state['cadastre'] = get_IGN_data(
            st.session_state['coords_bbox_Lambert'][0], 
            st.session_state['coords_bbox_Lambert'][1], 
            st.session_state['coords_bbox_Lambert'][2], 
            st.session_state['coords_bbox_Lambert'][3], 
            st.session_state['pixel_size'])
         st.session_state['fig'] = None

##############
# prédiction #
##############

# modèle YOLO  
@st.cache_resource
def getmodel_YOLO():
    return YOLO('models/YOLOv8_20240124_bruno.pt')
model_YOLO = getmodel_YOLO()

# paramètres des modèles
dict_models = {
   'YOLOv8' : {
      'predict_function' : predict_YOLOv8, 
      'model' : model_YOLO
   }
}

# taille en pixel
pixel_size = st.sidebar.slider('Taille (pixel)', min_value = PIXEL_SIZE_MIN, max_value = PIXEL_SIZE_MAX, value = st.session_state['pixel_size'], step = 100)
if pixel_size:
   st.session_state['pixel_size'] = pixel_size
   if all(st.session_state['coords_bbox_Lambert']):
      st.session_state['scale'] = (st.session_state['coords_bbox_Lambert'][1] - st.session_state['coords_bbox_Lambert'][0])/st.session_state['pixel_size']
      st.sidebar.caption('Echelle: {} m/pixel'.format(round(st.session_state['scale'], 1)))

# paramètres du modèle
seuil_conf = st.sidebar.slider('Seuil de confiance', min_value = 0.05, max_value = 0.95, value = 0.05, step = 0.05)
seuil_iou = st.sidebar.slider('Seuil IoU', min_value = 0.01, max_value = 0.99, value = 0.01, step = 0.01)
seuil_area = st.sidebar.slider('Seuil de surface', min_value = 0, max_value = 500, value = 10, step = 10)

# bouton de calcul
if calcul_button:
   if all((st.session_state['coords_bbox_Lambert'], 
          st.session_state['pixel_size'],
          st.session_state['scale'],
          st.session_state['orthophoto'],
          st.session_state['cadastre'])):
      with st.spinner('calcul de la prédiction ...'):
         st.session_state['fig'] = get_fig_prev(
            st.session_state['coords_bbox_Lambert'][0], 
            st.session_state['coords_bbox_Lambert'][1], 
            st.session_state['pixel_size'],
            st.session_state['scale'],
            st.session_state['orthophoto'],
            st.session_state['cadastre'])

# affichage de la prédiction
if st.session_state['fig'] is not None:
   st.plotly_chart(st.session_state['fig'])

