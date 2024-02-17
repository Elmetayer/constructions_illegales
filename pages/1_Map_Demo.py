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
    st.session_state['last_coords'] = [48.858370, 2.294481]
if 'last_clicked' not in st.session_state:
    st.session_state['last_clicked'] = None

def search_adresse():
    request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/search?q={}&index=address&limit=1&returntruegeometry=false'.format(
        st.session_state['adresse_text'])
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

def update_point():
    st.session_state['last_coords'] = st.session_state['last_clicked']
    st.session_state['last_clicked'] = None
    st.session_state['adresse_text'] = ''

# recherche de l'adresse dans la barre latérale
#with st.sidebar.form('adresse_search'):
adresse = st.sidebar.text_input('Adresse', key = 'adresse_text', on_change = search_adresse)
#    submit_adresse = st.form_submit_button('rechercher', on_click = search_adresse)

st.sidebar.write('coordonnées: ({}, {})'.format(
    st.session_state['last_clicked'][0], st.session_state['last_clicked'][1]))
st.sidebar.button('Mettre à jour', on_click = update_point)

# affichage de la carte et centrage sur l'adresse entrée
fg = folium.FeatureGroup(name = 'centre carte')
fg.add_child(folium.Marker(
    st.session_state['last_coords'], 
    popup = adresse, 
    tooltip = ''))
fg.add_child(folium.Marker(
    st.session_state['last_clicked'], 
    popup = adresse, 
    tooltip = ''))
m = folium.Map(location = CENTER_START, zoom_start = 16)
out_m = st_folium(m, feature_group_to_add = fg, center = st.session_state['last_coords'], width=725)
if out_m['last_clicked'] and st.session_state['last_clicked'] != [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]:
    st.session_state['last_clicked'] = [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]
    st.rerun()
