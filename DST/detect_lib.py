import plotly.graph_objs as go
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
import geopandas as gpd
from shapely.geometry import Polygon


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
