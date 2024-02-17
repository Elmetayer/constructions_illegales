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

# variables de session
CENTER_START = [48.858370, 2.294481]
if 'last_coords' not in st.session_state:
    st.session_state['last_coords'] = 'click'
if 'first_launch' not in st.session_state:
    st.session_state['first_launch'] = True

st.write('adresse courante: {}'.format(st.session_state['adresse_coords']))
st.write('coordonnées click courantes: {}'.format(st.session_state['click_coords']))
st.write('init: {}'.format(st.session_state['first_launch']))

# recherche de l'adresse dans la barre latérale
adresse = st.sidebar.text_input('Adresse', None)
if adresse:
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
    center, 
    popup = adresse, 
    tooltip = ''))
m = folium.Map(location = CENTER_START, zoom_start = 16)
out_m = st_folium(m, feature_group_to_add = fg, center = st.session_state['last_coords'], width=725)
if out_m['last_clicked'] and not st.session_state['first_launch']:
    if st.session_state['last_coords'] != [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]:
        st.session_state['last_coords'] = [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]
        st.rerun()
