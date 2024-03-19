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

# titre de la page
st.set_page_config(page_title="Détection", page_icon="👓")
st.markdown("# Détection")

# variables de session
PIXEL_SIZE_MAX = 1000
PIXEL_SCALE_REF = 0.2
SIZE_MAX = 1000
SIZE_YOLO = 512

if 'bbox_selected' not in st.session_state:
   if 'bbox' not in st.session_state:
      st.session_state['bbox_selected'] = None
   else:
      st.session_state['bbox_selected'] = st.session_state['bbox']
if 'coords_bbox_Lambert' not in st.session_state:
   if st.session_state['bbox_selected'] is None:
      st.session_state['coords_bbox_Lambert'] = (None, None, None, None)
   else:
      st.session_state['coords_bbox_Lambert'] = get_bbox_Lambert(st.session_state['bbox_selected'])
if 'bbox' not in st.session_state:
   st.session_state['refresh_bbox'] = 0
else:
   st.session_state['refresh_bbox'] = (st.session_state['bbox_selected'] != st.session_state['bbox'])*1
if 'pixel_size' not in st.session_state:
   if st.session_state['coords_bbox_Lambert'] != (None, None, None, None):
      coords_size = st.session_state['coords_bbox_Lambert'][1] - st.session_state['coords_bbox_Lambert'][0]
   else:
      coords_size = SIZE_MAX
   st.session_state['pixel_size'] = min(PIXEL_SIZE_MAX, int(coords_size/PIXEL_SCALE_REF))
if 'scale' not in st.session_state:
   if st.session_state['coords_bbox_Lambert'] != (None, None, None, None):
      st.session_state['scale'] = round((st.session_state['coords_bbox_Lambert'][1] - st.session_state['coords_bbox_Lambert'][0])/st.session_state['pixel_size'], 1)
   else:
      st.session_state['scale'] = None

# bouton de mise à jour
load_button = None
if st.session_state['refresh_bbox'] == 1:
   load_button = st.button('nouvelle zone')
if load_button:
   st.session_state['bbox_selected'] = st.session_state['bbox']
   st.session_state['coords_bbox_Lambert'] = get_bbox_Lambert(st.session_state['bbox_selected'])
   st.rerun()
   
# taille en pixel
pixel_size = st.sidebar.slider('Résolution (pixel)', 0, PIXEL_SIZE_MAX, st.session_state['pixel_size'], 100)
st.sidebar.caption('Echelle: {} m/pixel'.format(st.session_state['scale']))
if st.session_state['scale'] != PIXEL_SCALE_REF:
   st.sidebar.warning('attendion, l\'échelle de référence est {} m/pixel'.format(PIXEL_SCALE_REF))

# bouton de calcul
calcul_button = None
if st.session_state['pixel_size'] != pixel_size:
   calcul_button = st.sidebar.button('prédire')
if calcul_button:
   st.session_state['pixel_size'] = pixel_size
   if st.session_state['coords_bbox_Lambert'] != (None, None, None, None):
      st.session_state['scale'] = round((st.session_state['coords_bbox_Lambert'][1] - st.session_state['coords_bbox_Lambert'][0])/st.session_state['pixel_size'], 1)
   st.rerun()

##############
# prédiction #
##############

# modèle YOLO  
@st.cache_resource
def getmodel_YOLO():
    return YOLO('models/YOLOv8_20240124_bruno.pt')
model_YOLO = getmodel_YOLO()

# calcul de la prédiction
@st.cache_data(show_spinner = False)
def get_fig_prev(xmin, xmax, ymin, ymax, pixel_size, scale):
   if (xmin, xmax, ymin, ymax, pixel_size, scale) != (None, None, None, None, None, None):
      # ORTHOPHOTO
      request_wms = 'https://data.geopf.fr/wms-r?LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}'.format(
      xmin, ymin, xmax, ymax, pixel_size, pixel_size)
      response_wms = requests.get(request_wms).content
      orthophoto = Image.open(BytesIO(response_wms))
      # cadastre
      bounds = gpd.GeoDataFrame(
         {'Nom': ['name1', 'name2'],
         'geometry': [shapely.geometry.Point(xmin, ymin), shapely.geometry.Point(xmax, ymax)]},
         crs = 'EPSG:2154')
      bounds = bounds.to_crs('EPSG:4326')
      request_wfs = 'https://data.geopf.fr/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&typename=CADASTRALPARCELS.PARCELLAIRE_EXPRESS:batiment&outputformat=application/json&BBOX={},{},{},{}'.format(
    bounds.geometry[0].y, bounds.geometry[0].x, bounds.geometry[1].y, bounds.geometry[1].x)
      response_wfs = requests.get(request_wfs)
      gdf_cadastre = gpd.GeoDataFrame.from_features(response_wfs.json()['features'])
      if gdf_cadastre.shape[0]>0 :
         gdf_cadastre = gdf_cadastre.set_crs('EPSG:4326').to_crs('EPSG:2154')
         gdf_cadastre['geometry'] = gdf_cadastre['geometry'].make_valid()
         gdf_cadastre = gdf_cadastre.explode(index_parts = False)
         gdf_cadastre = gdf_cadastre[gdf_cadastre['geometry'].geom_type.isin(['Polygon', 'MultiPolygon'])]
      _, _, _, _, _, _, fig = affiche_contours(
         orthophoto, predict_YOLOv8, model_YOLO, SIZE_YOLO, 
         (xmin, ymin, scale), gdf_shapes_ref = gdf_cadastre,
         resolution_target = (pixel_size, pixel_size),
         seuil = 0.05, seuil_iou = 0.01, delta_only = False,
         seuil_area = 10,
         tolerance_polygone = 0.1)
      return fig, orthophoto, gdf_cadastre
   else:
      return None, None, None
with st.spinner('calcul de la prédiction ...'):
   fig, orthophoto, gdf_cadastre = get_fig_prev(
      st.session_state['coords_bbox_Lambert'][0], 
      st.session_state['coords_bbox_Lambert'][1], 
      st.session_state['coords_bbox_Lambert'][2], 
      st.session_state['coords_bbox_Lambert'][3], 
      st.session_state['pixel_size'],
      st.session_state['scale'])

# affichage de la prédiction
if fig is not None:
   st.plotly_chart(fig)
elif st.session_state['refresh_bbox'] != 1:
   st.write('aucun emplacement validé')
