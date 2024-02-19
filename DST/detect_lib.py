import plotly.graph_objs as go
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import geopandas as gpd
from shapely.geometry import Polygon

import streamlit as st
import folium
from streamlit_folium import st_folium
import shapely

# Fonction qui envoie une requete et récupère l'orthophoto dans un cadre rectangulaire (bounds en coord Lambert 93)
def charge_ortho(bounds):
    request = 'https://data.geopf.fr/wms-r?LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX='
    request += str(bounds.geometry[0].x)+","+str(bounds.geometry[0].y)+","+str(bounds.geometry[1].x)+","+str(bounds.geometry[1].y)
    request += '&WIDTH=1024&HEIGHT=1024'
    # print(request)
    response = requests.get(request).content
    orthophoto = Image.open(BytesIO(response))
    orthophoto = ImageOps.flip(orthophoto)
    return orthophoto

# Fonction qui envoie une requete et récupère la liste des bâtiments dans un cadre rectangulaire (bounds en coord Lambert 93)
def charge_batiments(bounds):
    bounds = bounds.to_crs("EPSG:4326")
    request = "https://wxs.ign.fr/parcellaire/geoportail/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&typename=CADASTRALPARCELS.PARCELLAIRE_EXPRESS:batiment&outputformat=application/json&bbox="
    request += str(bounds.geometry[0].y)+','+str(bounds.geometry[0].x)+','+str(bounds.geometry[1].y)+','+str(bounds.geometry[1].x)
    bounds = bounds.to_crs("EPSG:2154")
    response = requests.get(request)
    batiments = gpd.GeoDataFrame.from_features(response.json()["features"])
    if batiments.shape[0]>0 : 
        batiments = batiments.set_crs("EPSG:4326").to_crs("EPSG:2154")
        cadre = Polygon(((bounds.geometry[0].x, bounds.geometry[0].y), 
                        (bounds.geometry[1].x, bounds.geometry[0].y), 
                        (bounds.geometry[1].x, bounds.geometry[1].y), 
                        (bounds.geometry[0].x, bounds.geometry[1].y), 
                        (bounds.geometry[0].x, bounds.geometry[0].y)))
        batiments = batiments.intersection(cadre)
        batiments = batiments[~(batiments.geometry.isna() | batiments.geometry.is_empty)]
    return batiments

# Fonction personnalisée qui calcule l'IoU entre deux masques de même taille
def calcul_IoU(masque1, masque2, min_pixel=1):
    if not masque1.shape==masque2.shape:
        print("masques de tailles incompatibles")
        return None
    else:
        M1 = masque1 > 0
        M2 = masque2 > 0
        return (min_pixel+np.sum(M1 & M2))/(min_pixel+np.sum(M1 | M2))

# Fonction qui transforme une liste de shapes(listeX,listeY) en traces plotly pour affichage
def shape_to_traces(shapes, nom='traces', Centre=(0,0), echelle=1, couleur='blue', alpha=0.4):
    traces = []
    for i, (list_x, list_y) in enumerate(shapes):
        # Ajouter des points
        ligne = go.Scatter(
            x=(np.array(list_x)+Centre[0])/echelle,
            y=(np.array(list_y)+Centre[1])/echelle,
            line=dict(color=couleur, width=1),
            mode='lines',
            fill="toself",
            opacity=alpha,
            name=nom,
            legendgroup=nom,
            showlegend=(i==0)
        )
        traces.append(ligne)
    return traces

def search_adresse():
    '''
    fonction qui renvoie les coordonnées à partir d'une saisie d'adresse en texte libre
    utilise l'API de géocodage de l'IGN
    '''
    if st.session_state['adresse_field']:
        request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/search?q={}&index=address&limit=1&returntruegeometry=false'.format(
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
            st.session_state['new_adresse'] = adresses['features'][0]['properties']['label']
            st.session_state['adresse_field'] = ''
        else:
            st.session_state['warning_adresse'] = 'aucune adresse trouvée'

def search_lat_lon(lat_lon, defaut):
    '''
    fonction qui renvoie une adresse à partir de coordonnées
    utilise l'API de géocodage inversée de l'IGN
    '''
    result = defaut
    request_wxs = 'https://wxs.ign.fr/essentiels/geoportail/geocodage/rest/0.1/reverse?lat={}&lon={}&index=address&limit=1&returntruegeometry=false'.format(
        lat_lon[0], lat_lon[1])
    response_wxs = requests.get(request_wxs).content
    adresses = json.load(BytesIO(response_wxs))
    if len(adresses['features']) > 0:
        result = adresses['features'][0]['properties']['label']
    return result
    
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
    st.session_state['map_center'] = [
        (bbox_WSG.geometry[0].y + bbox_WSG.geometry[1].y)/2,
        (bbox_WSG.geometry[0].x + bbox_WSG.geometry[1].x)/2
    ]
    return(bbox_WSG.geometry[0].x, bbox_WSG.geometry[0].y, bbox_WSG.geometry[1].x, bbox_WSG.geometry[1].y)
