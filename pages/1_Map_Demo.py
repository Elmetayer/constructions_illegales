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
# convention pour la bbox : xmin, ymin, xmax, ymax
if 'bbox' not in st.session_state:
    st.session_state['bbox'] = None

st.write(st.session_state['bbox'])

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

def update_point():
    st.session_state['last_coords'] = st.session_state['last_clicked']
    st.session_state['adresse_text'] = ''

def get_bbox(coords_center, size, mode):
    ccoords_center_WSG = gpd.GeoDataFrame(
        {'Nom': ['centre'],
        'geometry': [shapely.geometry.Point(coords_center[0], coords_center[1])]},
        crs = 'EPSG:4326')
    coords_center_meter = ccoords_center_WSG.to_crs('EPSG:6933')
    if mode == 'haut/gauche':
        bbox_WSG = gpd.GeoDataFrame(
            {'Nom': ['min', 'max'],
            'geometry': [
                shapely.affinity.translate(shapely.geometry.Point(ccoords_center_WSG.geometry[0].x, ccoords_center_WSG.geometry[0].x), 0, size),
                shapely.affinity.translate(shapely.geometry.Point(ccoords_center_WSG.geometry[0].y, ccoords_center_WSG.geometry[0].y), -size, 0)]},
            crs = 'EPSG:4087')
    if mode == 'centre':
        bbox_meters = gpd.GeoDataFrame(
            {'Nom': ['min', 'max'],
            'geometry': [
                shapely.geometry.Point(coords_center_meter.geometry[0].x - size//2, coords_center_meter.geometry[0].y - size//2),
                shapely.geometry.Point(coords_center_meter.geometry[0].x + size//2, coords_center_meter.geometry[0].y + size//2)]},
            crs = 'EPSG:6933')
        bbox_WSG = bbox_meters.to_crs('EPSG:4326')
    return(bbox_WSG.geometry[0].x, bbox_WSG.geometry[0].y, bbox_WSG.geometry[1].x, bbox_WSG.geometry[1].y)


# mode d'affichage et taille de la bouding box
bbox_mode = st.sidebar.radio('Bounding box', ['haut/gauche', 'centre'])
bbox_size = st.sidebar.slider('Taille (m)', 0, 500, 100)
if bbox_mode:
    st.session_state['bbox'] = get_bbox(st.session_state['last_clicked'], bbox_size, bbox_mode)
if bbox_size:
    st.session_state['bbox'] = get_bbox(st.session_state['last_clicked'], bbox_size, bbox_mode)

# recherche de l'adresse dans la barre latérale
adresse = st.sidebar.text_input('Adresse', key = 'adresse_text', on_change = search_adresse)

# gestion des points de recherche
update_button = st.sidebar.button('valider le point', on_click = update_point)
cancel_button = st.sidebar.button('annuler le point')
if cancel_button:
    st.session_state['last_clicked'] = None
    st.rerun()

# affichage de la carte et centrage sur l'adresse entrée
fg = folium.FeatureGroup(name = 'centre carte')

style_bbox = {
    'color': '#ff3939',
    'fillOpacity': 0,
    'weight': 3,
    'opacity': 1,
    'dashArray': '5, 5'}

if st.session_state['last_clicked']:
    # pointeur
    fg.add_child(folium.Marker(
        st.session_state['last_clicked'], 
        popup = st.session_state['last_clicked'], 
        tooltip = st.session_state['last_clicked']))
if st.session_state['bbox']:
    # bounding box
    polygon_bbox = shapely.Polygon((
        (st.session_state['bbox'][0], st.session_state['bbox'][0]), 
        (st.session_state['bbox'][1], st.session_state['bbox'][0]), 
        (st.session_state['bbox'][1], st.session_state['bbox'][1]),
        (st.session_state['bbox'][0], st.session_state['bbox'][1])))
    gdf_bbox = gpd.GeoDataFrame(geometry = [polygon_bbox]).set_crs(epsg = 4326)
    polygon_folium_bbox = folium.GeoJson(data = gdf_bbox, style_function = lambda x: style_bbox)
    fg.add_child(polygon_folium_bbox)

m = folium.Map(location = CENTER_START, zoom_start = 16)
out_m = st_folium(
    m, 
    feature_group_to_add = fg, 
    center = st.session_state['last_coords'], 
    width = 700,
    height = 700)
if out_m['last_clicked'] and st.session_state['last_clicked'] != [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]:
    st.session_state['last_clicked'] = [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]
    st.session_state['bbox'] = get_bbox(st.session_state['last_clicked'], bbox_size, bbox_mode)
    st.rerun()
