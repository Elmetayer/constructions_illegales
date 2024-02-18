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


# titre de la page
st.set_page_config(page_title="Display Demo", page_icon="👓")
st.markdown("# Display Demo")
st.sidebar.header("Display Demo")

# variables de session
PIXEL_SIZE_MAX = 1000
PIXEL_SCALE_REF = 0.2

# calcul de l'image
coords_bbox_WSG = gpd.GeoDataFrame(
   {'Nom': ['min', max],
   'geometry': [
      shapely.geometry.Point(st.session_state['bbox'][0], st.session_state['bbox'][1]),
      shapely.geometry.Point(st.session_state['bbox'][2], st.session_state['bbox'][3])]},
   crs = 'EPSG:4326')
ccoords_bbox_Lambert = coords_bbox_WSG.to_crs('EPSG:2154')
xmin = ccoords_bbox_Lambert.geometry[0].x
xmax = ccoords_bbox_Lambert.geometry[1].x
ymin = ccoords_bbox_Lambert.geometry[0].y
ymax = ccoords_bbox_Lambert.geometry[1].y

# taille en pixel
pixel_size_defaut = min(PIXEL_SIZE_MAX, int((xmax - xmin)/PIXEL_SCALE_REF))
pixel_size = st.sidebar.slider('Taille (pixel)', 0, PIXEL_SIZE_MAX, pixel_size_defaut, 100)
scale = round(pixel_size/(xmax - xmin), 1)
st.sidebar.caption('Echelle: {} pixel/m'.format(scale))
if scale != PIXEL_SCALE_REF:
    st.sidebar.warning('attendion, l\'échelle de référence est {}'.format(PIXEL_SCALE_REF))

# récupération et affichage de l'orthophoto
request_wms = 'https://data.geopf.fr/wms-r?LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}'.format(
   xmin, ymin, xmax, ymax, pixel_size, pixel_size)
response_wms = requests.get(request_wms).content
orthophoto = Image.open(BytesIO(response_wms))

fig = px.imshow(orthophoto, width = 600, height = 600)
st.plotly_chart(fig)
