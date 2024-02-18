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
PIXEL_SCALE = 0.2
PIXEL_SIZE_DEFAUT = 1000

# taille en pixel
pixel_size = st.sidebar.slider('Taille (pixel)', 0, 1000, PIXEL_SIZE_DEFAUT, 100)
st.sidebar.legend('Taille (m): '.format(PIXEL_SCALE*pixel_size))

# calcul de l'image
coords_bbox_WSG = gpd.GeoDataFrame(
   {'Nom': ['min', max],
   'geometry': [
      shapely.geometry.Point(st.session_state['bbox'][0], st.session_state['bbox'][1]),
      shapely.geometry.Point(st.session_state['bbox'][2], st.session_state['bbox'][3])]},
   crs = 'EPSG:4326')
ccoords_bbox_Lambert = coords_bbox_WSG.to_crs('EPSG:2154')
X0 = ccoords_bbox_Lambert.geometry[0].x
Y0 = ccoords_bbox_Lambert.geometry[1].y

raster_transform = rasterio.transform.Affine(PIXEL_SCALE, 0.0, X0,
                          0.0, -PIXEL_SCALE, Y0 + PIXEL_SCALE*pixel_size)
xmin, ymax = raster_transform*(0, 0)
xmax, ymin = raster_transform*(pixel_size, pixel_size)
request_wms = 'https://data.geopf.fr/wms-r?LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}'.format(
   xmin, ymin, xmax, ymax, pixel_size, pixel_size)
response_wms = requests.get(request_wms).content
orthophoto = Image.open(BytesIO(response_wms))

fig = px.imshow(orthophoto)
st.plotly_chart(fig)
