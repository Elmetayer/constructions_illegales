import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
import requests
from io import BytesIO
import geopandas as gpd
from PIL import Image
from ultralytics import YOLO
from keras.models import load_model

from pages.functions.functions import *
from pages.functions.yolo import *
from pages.functions.unet import *

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
      st.session_state['scale'] = (st.session_state['coords_bbox_Lambert'][2] - st.session_state['coords_bbox_Lambert'][0])/st.session_state['pixel_size']
   else:
      st.session_state['scale'] = None
if 'fig' not in st.session_state:
   st.session_state['fig'] = None
if 'cadastre' not in st.session_state:
   st.session_state['cadastre'] = None
if 'orthophoto' not in st.session_state:
   st.session_state['orthophoto'] = None

#################
# données IGN #
#################
   
container_IGN = st.sidebar.container(border=True)
   
# taille en pixel
with container_IGN:
   pixel_size = st.slider('Taille (pixel)', min_value = PIXEL_SIZE_MIN, max_value = PIXEL_SIZE_MAX, value = st.session_state['pixel_size'], step = 100)
if pixel_size:
   st.session_state['pixel_size'] = pixel_size
   if all(st.session_state['coords_bbox_Lambert']):
      st.session_state['scale'] = (st.session_state['coords_bbox_Lambert'][2] - st.session_state['coords_bbox_Lambert'][0])/st.session_state['pixel_size']
      with container_IGN:
         st.caption('Echelle: {} m/pixel'.format(round(st.session_state['scale'], 1)))

# chargement des données
with container_IGN:
   load_button = st.button('données IGN')
if load_button:
   if 'bbox' in st.session_state:
      st.session_state['bbox_selected'] = st.session_state['bbox']
   if all((st.session_state['bbox_selected'], st.session_state['pixel_size'])):
      st.session_state['coords_bbox_Lambert'] = get_bbox_Lambert(st.session_state['bbox_selected'])
      with st.spinner('récupération des données IGN ...'):
         # @st.cache_data(show_spinner = False)
         def get_IGN_data(xmin, ymin, xmax, ymax, pixel_size):
            if all((xmin, ymin, xmax, ymax, pixel_size)):
               request_wms = 'https://data.geopf.fr/wms-r?LAYERS=HR.ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}'.format(
               xmin, ymin, xmax, ymax, pixel_size, pixel_size)
               response_wms = requests.get(request_wms).content
               orthophoto = Image.open(BytesIO(response_wms))
               bounds = gpd.GeoDataFrame(
                  {'Nom': ['min', 'max'],
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
               return orthophoto, gdf_cadastre
            else:
               return None, None
         st.session_state['orthophoto'], st.session_state['cadastre'] = get_IGN_data(
            st.session_state['coords_bbox_Lambert'][0], 
            st.session_state['coords_bbox_Lambert'][1], 
            st.session_state['coords_bbox_Lambert'][2], 
            st.session_state['coords_bbox_Lambert'][3], 
            st.session_state['pixel_size'])
      with st.spinner('affichage des données IGN ...'):
         @st.cache_data(show_spinner = False)
         def get_fig_IGN(X0, YO, pixel_size, scale, _orthophoto, _gdf_cadastre):
            if all((X0, YO, pixel_size, scale, _orthophoto, _gdf_cadastre is not None)):
               _, _, _, _, _, _, fig = affiche_contours(
                  _orthophoto, None, None, None, 
                  (X0, YO, scale), gdf_shapes_ref = _gdf_cadastre,
                  resolution_target = (pixel_size, pixel_size),
                  seuil = None, seuil_iou = None,
                  seuil_area = None,
                  tolerance_polygone = None)
               return fig
            else:
               return None
         st.session_state['fig'] = get_fig_IGN(
            st.session_state['coords_bbox_Lambert'][0], 
            st.session_state['coords_bbox_Lambert'][1], 
            st.session_state['pixel_size'],
            st.session_state['scale'],
            st.session_state['orthophoto'],
            st.session_state['cadastre'])
   else:
      st.write('⚠️ zone non sélectionnée')

##############
# prédiction #
##############
      
container_pred = st.sidebar.container(border=True)

# modèle YOLO  
@st.cache_resource
def getmodel_YOLO():
    return YOLO('models/YOLOv8_20240124_bruno.pt')
model_YOLO = getmodel_YOLO()

# modèle Unet  
@st.cache_resource
def getmodel_Unet():
    return load_model('models/UNet07_res512_23_12_23.h5')
model_Unet = getmodel_Unet()

# paramètres des modèles
dict_models = {
   'YOLOv8' : {
      'predict_function' : predict_YOLOv8, 
      'model' : model_YOLO,
      'size' : 512
   },
   'Unet' : {
      'predict_function' : predict_Unet, 
      'model' : model_Unet,
      'size' : (512, 512)
   }
}

# paramètres du modèle
with container_pred:
   seuil_conf = st.slider('Seuil de confiance', min_value = 0.05, max_value = 0.95, value = 0.05, step = 0.05)
   seuil_iou = st.slider('Seuil IoU', min_value = 0.01, max_value = 0.99, value = 0.01, step = 0.01)
   seuil_area = st.slider('Seuil de surface', min_value = 0, max_value = 500, value = 10, step = 10)
   model_predict = st.selectbox('modèle', dict_models.keys())

# bouton de calcul
with container_pred:
   calcul_button = st.button('prédire')
if calcul_button:
   if all((st.session_state['coords_bbox_Lambert'], 
          st.session_state['pixel_size'],
          st.session_state['scale'],
          st.session_state['orthophoto'],
          st.session_state['cadastre'] is not None)):
      with st.spinner('calcul de la prédiction ...'):
         @st.cache_data(show_spinner = False)
         def get_fig_prev(X0, YO, pixel_size, scale, _orthophoto, _gdf_cadastre, model_predict, seuil_conf, seuil_iou, seuil_area):
            if all((X0, YO, pixel_size, scale, _orthophoto, _gdf_cadastre is not None)):
               _, _, _, _, _, _, fig = affiche_contours(
                  _orthophoto, dict_models[model_predict]['predict_function'], dict_models[model_predict]['model'], dict_models[model_predict]['size'], 
                  (X0, YO, scale), gdf_shapes_ref = _gdf_cadastre,
                  resolution_target = (pixel_size, pixel_size),
                  seuil = seuil_conf, seuil_iou = seuil_iou,
                  seuil_area = seuil_area,
                  tolerance_polygone = 0.1)
               return fig
            else:
               return None
         st.session_state['fig'] = get_fig_prev(
            st.session_state['coords_bbox_Lambert'][0], 
            st.session_state['coords_bbox_Lambert'][1], 
            st.session_state['pixel_size'],
            st.session_state['scale'],
            st.session_state['orthophoto'],
            st.session_state['cadastre'],
            model_predict,
            seuil_conf, seuil_iou, seuil_area)
   else:
      st.write('⚠️ données IGN absentes')

# affichage de la prédiction
if st.session_state['fig'] is not None:
   st.plotly_chart(st.session_state['fig'], use_container_width = True)

