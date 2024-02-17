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
    if st.session_state['adresse_text']:
        request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/search?q={}&index=address&limit=1&returntruegeometry=false'.format(
            st.session_state['adresse_text'])
        response_wxs = requests.get(request_wxs).content
        adresses = json.load(BytesIO(response_wxs))
        if len(adresses['features']) > 0:
            X0 = adresses['features'][0]['properties']['x']
            Y0 = adresses['features'][0]['properties']['y']
            coords_Lambert = gpd.GeoDataFrame(
                {'Nom': ['adresse'],
                 'geometry': [shapely.geometry.Point(X0, Y0)]},
                crs = 'EPSG:2154')
            coords_WSG = coords_Lambert.to_crs('EPSG:4326')
            st.session_state['last_coords'] = [coords_WSG.geometry[0].y, coords_WSG.geometry[0].x]
            st.session_state['adresse_text'] = adresses['features'][0]['properties']['label']
    
# recherche de l'adresse dans la barre latérale
adresse = st.sidebar.text_input('Adresse', key = 'adresse_text', on_change = search_adresse)

# gestion des points de recherche
update_button = st.sidebar.button('valider le point')
if update_button:
    st.session_state['last_coords'] = st.session_state['last_clicked']
    st.session_state['adresse_text'] = ''
    st.rerun()
cancel_button = st.sidebar.button('annuler le point')
if cancel_button:
    st.session_state['last_clicked'] = None
    st.rerun()

# affichage de la carte et centrage sur l'adresse entrée
fg = folium.FeatureGroup(name = 'centre carte')
if st.session_state['last_clicked'] and st.session_state['last_clicked'] != st.session_state['last_coords']:
    fg.add_child(folium.Marker(
        st.session_state['last_clicked'], 
        popup = st.session_state['last_clicked'], 
        tooltip = st.session_state['last_clicked']))
m = folium.Map(location = CENTER_START, zoom_start = 16)
out_m = st_folium(m, feature_group_to_add = fg, center = st.session_state['last_coords'], width=725)
if out_m['last_clicked'] and st.session_state['last_clicked'] != [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]:
    st.session_state['last_clicked'] = [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]
    st.rerun()
