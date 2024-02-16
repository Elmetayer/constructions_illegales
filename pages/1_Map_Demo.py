import folium
import streamlit as st
import request
import json
import shapely
import geopandas
from streamlit_folium import st_folium

# titre de la page
st.set_page_config(page_title="Map Demo", page_icon="📈")
st.markdown("# Map Demo")
st.sidebar.header("Map Demo")

# recherche de l'adresse dans la barre latérale
title = st.side_bar.text_input('Adresse', '')
request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/search?q={}&index=address&limit=1&returntruegeometry=false'.format(
    adresse
)
response_wxs = requests.get(request_wxs).content
adresses = json.load(BytesIO(response_wxs))
X0 = adresses['features'][0]['properties']['x']
Y0 = adresses['features'][0]['properties']['y']
coords = gpd.GeoDataFrame(
    {'Nom': ['adresse'],
     'geometry': [shapely.geometry.Point(X0, YO)]},
    crs = 'EPSG:2154')
bounds = bounds.to_crs('EPSG:4326')
st.write('Les coordonnées sont: ({}, {})'.format(coords.geometry[0].x, coords.geometry[0].y))

# center on Liberty Bell, add marker
m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
folium.Marker(
    [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
).add_to(m)

# call to render Folium map in Streamlit
st_data = st_folium(m, width=725)