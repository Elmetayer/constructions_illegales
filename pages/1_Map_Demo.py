import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import json
from io import BytesIO
import shapely
import geopandas as gpd

def search_adresse():
    '''
    fonction qui renvoie les coordonnées à partir d'une saisie d'adresse en texte libre
    utilise l'API de géocodage de l'IGN
    '''
    if st.session_state['adresse_field']:
        '''
        ancienne URL IGN
        request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/search?q={}&index=address&limit=1&returntruegeometry=false'.format(
            st.session_state['adresse_field'])
        '''
        request_wxs = 'https://data.geopf.fr/geocodage/rest/0.1/search?q={}&index=address&limit=1&returntruegeometry=false'.format(
            st.session_state['adresse_field'])
        response_wxs = requests.get(request_wxs).content
        adresses = json.load(BytesIO(response_wxs))
        if len(adresses['features']) > 0:
            st.session_state['warning_adresse'] = None
            X0 = adresses['features'][0]['properties']['x']
            Y0 = adresses['features'][0]['properties']['y']
            coords_Lambert = gpd.GeoDataFrame(
                {'Nom': ['adresse'],
                 'geometry': [shapely.geometry.Point(X0, Y0)]},
                crs = 'EPSG:2154')
            coords_WSG = coords_Lambert.to_crs('EPSG:4326')
            st.session_state['new_point'] = [coords_WSG.geometry[0].y, coords_WSG.geometry[0].x]
            st.session_state['map_center'] = st.session_state['new_point']
            st.session_state['new_adresse'] = adresses['features'][0]['properties']['label']
            st.session_state['adresse_field'] = ''
        else:
            st.session_state['warning_adresse'] = 'aucune adresse trouvée'

def search_lat_lon(lat_lon):
    '''
    fonction qui renvoie une adresse à partir de coordonnées
    utilise l'API de géocodage inversée de l'IGN
    '''
    result = ADRESSE_DEFAUT
    '''
    ancienne URL IGN
    request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/reverse?lat={}&lon={}&index=address&limit=1&returntruegeometry=false'.format(
        lat_lon[0], lat_lon[1])
    '''
    request_wxs = 'https://data.geopf.fr/geocodage/rest/0.1/reverse?lat={}&lon={}&index=address&limit=1&returntruegeometry=false'.format(
        lat_lon[0], lat_lon[1])
    response_wxs = requests.get(request_wxs).content
    adresses = json.load(BytesIO(response_wxs))
    if len(adresses['features']) > 0:
        result = adresses['features'][0]['properties']['label']
    return result

def update_point():
    '''
    fonction qui met à jour le point valide
    '''
    if st.session_state['new_point']:
        st.session_state['last_coords'] = st.session_state['new_point']
        st.session_state['adresse_text'] = st.session_state['new_adresse']
        st.session_state['new_point'] = None
        st.session_state['new_adresse'] = ADRESSE_DEFAUT
        st.session_state['bbox'] = get_bbox(st.session_state['last_coords'], bbox_size, bbox_mode)
        st.session_state['map_center'] = get_bbox_center(st.session_state['bbox'])
    
def get_bbox(coords_center, size, mode):
    '''
    fonction qui calcule les coordonnées xmin, ymin, xmax, ymax de la bounding box
    à partir du point de référence, de la taille et du mode
    '''
    ccoords_center_WSG = gpd.GeoDataFrame(
        {'Nom': ['centre'],
        'geometry': [shapely.geometry.Point(coords_center[1], coords_center[0])]},
        crs = 'EPSG:4326')
    coords_center_Lambert = ccoords_center_WSG.to_crs('EPSG:2154')
    if mode == 'haut/gauche':
        bbox_Lambert = gpd.GeoDataFrame(
            {'Nom': ['min', 'max'],
            'geometry': [
                shapely.geometry.Point(coords_center_Lambert.geometry[0].x, coords_center_Lambert.geometry[0].y - size),
                shapely.geometry.Point(coords_center_Lambert.geometry[0].x + size, coords_center_Lambert.geometry[0].y)]},
            crs = 'EPSG:2154')
    if mode == 'centre':
        bbox_Lambert = gpd.GeoDataFrame(
            {'Nom': ['min', 'max'],
            'geometry': [
                shapely.geometry.Point(coords_center_Lambert.geometry[0].x - size//2, coords_center_Lambert.geometry[0].y - size//2),
                shapely.geometry.Point(coords_center_Lambert.geometry[0].x + size//2, coords_center_Lambert.geometry[0].y + size//2)]},
            crs = 'EPSG:2154')
    bbox_WSG = bbox_Lambert.to_crs('EPSG:4326')
    return(bbox_WSG.geometry[0].x, bbox_WSG.geometry[0].y, bbox_WSG.geometry[1].x, bbox_WSG.geometry[1].y)

def get_bbox_center(bbox):
    '''
    renvoie le centre de la bounding box
    '''
    return([(bbox[1] + bbox[3])/2, (bbox[0] + bbox[2])/2])

# titre de la page
st.set_page_config(page_title="Map Demo", page_icon="🔎")
st.markdown("# Map Demo")

# variables de session
CENTER_START = [48.858370, 2.294481]
ADRESSE_DEFAUT = 'non defini'
SIZE_DEFAUT = 500
MODE_DEFAUT = 'haut/gauche'
ZOOM_DEFAUT = 14
EPSILON_COORD = 0.00001

if 'last_coords' not in st.session_state:
    st.session_state['last_coords'] = [48.858370, 2.294481]
# convention pour la bbox : xmin, ymin, xmax, ymax
if 'bbox' not in st.session_state:
    st.session_state['bbox'] = get_bbox(st.session_state['last_coords'], SIZE_DEFAUT, MODE_DEFAUT)
if 'map_center' not in st.session_state:
    st.session_state['map_center'] = get_bbox_center(st.session_state['bbox'])
if 'adresse_text' not in st.session_state:
    st.session_state['adresse_text'] = ADRESSE_DEFAUT
if 'new_point' not in st.session_state:
    st.session_state['new_point'] = None
if 'new_adresse' not in st.session_state:
    st.session_state['new_adresse'] = ADRESSE_DEFAUT
if 'warning_adresse' not in st.session_state:
    st.session_state['warning_adresse'] = None    
if 'last_clicked' not in st.session_state:
    st.session_state['last_clicked'] = None
if 'bbox_size' not in st.session_state:
    st.session_state['bbox_size'] = SIZE_DEFAUT

# fond de carte
satellite = st.sidebar.checkbox('satellite', False)

# mode d'affichage et taille de la bouding box
bbox_mode = st.sidebar.radio('Bounding box', [MODE_DEFAUT, 'centre'], horizontal = True)
bbox_size = st.sidebar.slider('Taille (m)', 0, 1000, st.session_state['bbox_size'])
if bbox_mode:
    st.session_state['bbox'] = get_bbox(st.session_state['last_coords'], bbox_size, bbox_mode)
if bbox_size:
    st.session_state['bbox_size'] = bbox_size
    st.session_state['bbox'] = get_bbox(st.session_state['last_coords'], bbox_size, bbox_mode)
    
# recherche de l'adresse dans la barre latérale
adresse = st.sidebar.text_input('Adresse', key = 'adresse_field', on_change = search_adresse, placeholder = 'entrer une adresse', label_visibility = 'collapsed')
if st.session_state['warning_adresse']:
    st.sidebar.warning(st.session_state['warning_adresse'])

# gestion des points de recherche
update_button = None
cancel_button = None
if st.session_state['new_point']:
    col1, col2 = st.sidebar.columns([1,1])
    with col1:
        update_button = st.button('valider le point', on_click = update_point)
    with col2:
        cancel_button = st.button('annuler le point')
if cancel_button:
    st.session_state['new_point'] = None
    st.session_state['adresse_clicked'] = ADRESSE_DEFAUT
    st.session_state['map_center'] = get_bbox_center(st.session_state['bbox'])
    # astuce pour provoquer le rafraîchissement
    st.session_state['map_center'] = [st.session_state['map_center'][0]+EPSILON_COORD, st.session_state['map_center'][1]+EPSILON_COORD]
    st.rerun()

# affichage de la carte et centrage sur l'adresse entrée

center_button = st.button('centrer la carte')
if center_button:
    st.session_state['map_center'] = get_bbox_center(st.session_state['bbox'])
    # astuce pour provoquer le rafraîchissement
    st.session_state['map_center'] = [st.session_state['map_center'][0]+EPSILON_COORD, st.session_state['map_center'][1]+EPSILON_COORD]
    
fg = folium.FeatureGroup(name = 'centre carte')

style_bbox = {
    'color': '#ff3939',
    'fillOpacity': 0,
    'weight': 3,
    'opacity': 1,
    'dashArray': '5, 5'}

# point courant
fg.add_child(folium.Marker(
    st.session_state['last_coords'], 
    popup = st.session_state['adresse_text'], 
    tooltip = st.session_state['last_coords']))
if st.session_state['new_point']:
    # pointeur
    fg.add_child(folium.Marker(
        st.session_state['new_point'], 
        popup = st.session_state['new_adresse'], 
        tooltip = st.session_state['new_point']))
if st.session_state['bbox']:
    # bounding box
    polygon_bbox = shapely.Polygon((
        (st.session_state['bbox'][0], st.session_state['bbox'][1]), 
        (st.session_state['bbox'][2], st.session_state['bbox'][1]), 
        (st.session_state['bbox'][2], st.session_state['bbox'][3]),
        (st.session_state['bbox'][0], st.session_state['bbox'][3])))
    gdf_bbox = gpd.GeoDataFrame(geometry = [polygon_bbox]).set_crs(epsg = 4326)
    polygon_folium_bbox = folium.GeoJson(data = gdf_bbox, style_function = lambda x: style_bbox)
    fg.add_child(polygon_folium_bbox)

m = folium.Map(location = CENTER_START, zoom_start = ZOOM_DEFAUT)
if satellite:
    tile = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = False,
            control = True).add_to(m)
out_m = st_folium(
    m, 
    feature_group_to_add = fg, 
    center = st.session_state['map_center'], 
    width = 800,
    height = 800)
if out_m['last_clicked'] and st.session_state['last_clicked'] != [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]:
    st.session_state['last_clicked'] = [out_m['last_clicked']['lat'], out_m['last_clicked']['lng']]
    st.session_state['new_point'] = st.session_state['last_clicked']
    st.session_state['new_adresse'] = search_lat_lon(st.session_state['new_point'])
    st.rerun()

    
