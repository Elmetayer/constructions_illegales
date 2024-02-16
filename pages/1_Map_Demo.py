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
st.session_state['last_coords'] = [coords_WSG.geometry[0].y, coords_WSG.geometry[0].x]

# affichage de la carte et centrage sur l'adresse entrée
fg = folium.FeatureGroup(name = 'centre carte')
fg.add_child(folium.Marker(
    st.session_state['last_coords'], 
    popup = adresse, 
    tooltip = ''))
m = folium.Map(location = st.session_state['last_coords'], zoom_start = 16)
out_m = st_folium(m, feature_group_to_add = fg, width=725)
if (out_m['last_clicked'] and out_m['last_clicked'] != st.session_state['last_coords']):
    st.session_state['last_coords'] = out_m['last_clicked']
    st.experimental_rerun()
