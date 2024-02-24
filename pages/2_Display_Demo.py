import plotly.express as px
import rasterio
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json
from io import BytesIO
import shapely
import geopandas as gpd
from PIL import Image, ImageOps

def get_bbox_Lambert(bbox):
   '''
   fonction qui renvoie un gdf en Lambert à partir d'une bbox en WSG84
   '''
   coords_bbox_WSG = gpd.GeoDataFrame({
      'Nom': ['min', max],
      'geometry': [
         shapely.geometry.Point(bbox[0], bbox[1]),
         shapely.geometry.Point(bbox[2], bbox[3])]},
      crs = 'EPSG:4326')
   return(coords_bbox_WSG.to_crs('EPSG:2154'))

def load():
    '''
    fonction qui met à jour la bbox courante
    '''
    st.session_state['bbox_selected'] = st.session_state['bbox']
    st.session_state['coords_bbox_Lambert'] = get_bbox_Lambert(st.session_state['bbox_selected'])

# titre de la page
st.set_page_config(page_title="Display Demo", page_icon="👓")
st.markdown("# Display Demo")
st.sidebar.header("Display Demo")

# variables de session
PIXEL_SIZE_MAX = 1000
PIXEL_SCALE_REF = 0.2
SIZE_MAX = 1000

if 'bbox' not in st.session_state:
   st.session_state['bbox'] = None
if 'bbox_selected' not in st.session_state:
   st.session_state['bbox_selected'] = st.session_state['bbox']
if 'coords_bbox_Lambert' not in st.session_state:
   if st.session_state['bbox_selected'] is None:
      st.session_state['coords_bbox_Lambert'] = None
   else:
      st.session_state['coords_bbox_Lambert'] = get_bbox_Lambert(st.session_state['bbox_selected'])

# bouton de mise à jour
load_button = None
if st.session_state['bbox_selected'] != st.session_state['bbox']:
    load_button = st.button('mettre à jour', on_click = load)

# taille en pixel
if st.session_state['coords_bbox_Lambert']:
   coords_size = st.session_state['coords_bbox_Lambert'].geometry[1].x - st.session_state['coords_bbox_Lambert'].geometry[0].x
else:
   coords_size = SIZE_MAX
pixel_size_defaut = min(PIXEL_SIZE_MAX, int(coords_size/PIXEL_SCALE_REF))
pixel_size = st.sidebar.slider('Taille (pixel)', 0, PIXEL_SIZE_MAX, pixel_size_defaut, 100)
scale = round(coords_size/pixel_size, 1)
st.sidebar.caption('Echelle: {} m/pixel'.format(scale))
if scale != PIXEL_SCALE_REF:
    st.sidebar.warning('attendion, l\'échelle de référence est {} m/pixel'.format(PIXEL_SCALE_REF))

# récupération et affichage de l'orthophoto
@st.cache_data
def get_fig_ortho_cached(coords_bbox_Lambert, pixel_size):
   if ccoords_bbox_Lambert is not None:
      xmin = coords_bbox_Lambert.geometry[0].x
      xmax = coords_bbox_Lambert.geometry[1].x
      ymin = coords_bbox_Lambert.geometry[0].y
      ymax = coords_bbox_Lambert.geometry[1].y
      request_wms = 'https://data.geopf.fr/wms-r?LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}'.format(
         xmin, ymin, xmax, ymax, pixel_size, pixel_size)
      response_wms = requests.get(request_wms).content
      orthophoto = Image.open(BytesIO(response_wms))
      fig = px.imshow(orthophoto, width = 800, height = 800)
      return(fig)
   else:
      return(None)
fig = get_fig_ortho_cached(st.session_state['coords_bbox_Lambert'], pixel_size)
if fig is not None:
   st.plotly_chart(fig)
