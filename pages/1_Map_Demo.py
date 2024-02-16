import folium
import streamlit as st
import requests
import json
import shapely
import geopandas
from streamlit_folium import st_folium
from io import BytesIO
import geopandas as gpd

# titre de la page
st.set_page_config(page_title="Map Demo", page_icon="📈")
st.markdown("# Map Demo")
st.sidebar.header("Map Demo")

# recherche de l'adresse dans la barre latérale
adresse = st.sidebar.text_input('Adresse', 'Champ de Mars, 5 Av. Anatole France, 75007 Paris')
request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/search?q={}&index=address&limit=1&returntruegeometry=false'.format(
    adresse)
response_wxs = requests.get(request_wxs).content
adresses = json.load(BytesIO(response_wxs))
X0 = adresses['features'][0]['properties']['x']
Y0 = adresses['features'][0]['properties']['y']
coords_Lambert = gpd.GeoDataFrame(
    {'Nom': ['adresse'],
     'geometry': [shapely.geometry.Point(X0, Y0)]},
    crs = 'EPSG:2154')
coords_WSG = coords_Lambert.to_crs('EPSG:4326')
st.write('Les coordonnées sont: ({}, {})'.format(coords_WSG.geometry[0].x, coords_WSG.geometry[0].y))

# center on Liberty Bell, add marker
m = folium.Map(location=[coords_WSG.geometry[0].y, coords_WSG.geometry[0].x], zoom_start=16)
folium.Marker(
    [coords_WSG.geometry[0].y, coords_WSG.geometry[0].x], 
    popup = adresse, 
    tooltip = adresse).add_to(m)

# call to render Folium map in Streamlit
st_data = st_folium(m, width=725)
st.write(json.load(st_data))