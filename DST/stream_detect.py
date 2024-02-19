import streamlit as st
# import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd
# from PIL import Image, ImageOps
# from io import BytesIO
import rasterio.features
import rasterio.transform
import plotly.express as px
import plotly.graph_objs as go
#from ultralytics import YOLO
import cv2

import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
import keras

from detect_lib import *

# Zone utilisée pour l'entrainement
# Xbounds = [825000.0, 844800.0]
# Ybounds = [6505000.0, 6519800.0]

# Zone plus étendue
Xbounds = [750000.0, 900000.0]
Ybounds = [6400000.0, 6800000.0]
chemin_model = 'DST/Moez-UNet07_res512_23_12_23.h5'

# Streamlit
st.title("Projet de détection des bâtiments")
st.sidebar.title("Sommaire")
pages=["Chargement de la carte", "Modélisation", "Détection des bâtiments"]
page=st.sidebar.radio("Aller vers", pages)

if page == pages[0] : 
    st.header("Visualisation de la zone d'étude")
if page == pages[1] : 
    st.header("Modélisation")

c = st.container(border=True)
X0 = c.slider("X Lambert",Xbounds[0],Xbounds[1],step=200.0)
Y0 = c.slider("Y Lambert",Ybounds[0],Ybounds[1],step=200.0)

d = {'Nom': ['name1', 'name2'], 'geometry': [Point(X0-204.8, Y0-204.8), Point(X0+204.8, Y0+204.8)]}
bounds = gpd.GeoDataFrame(d, crs="EPSG:2154")
# st.write(bounds)
# st.write(bounds.crs)

## ---- Chargement de l'orthophoto
orthophoto = charge_ortho(bounds)

## ---- Chargement du cadastre
batiments = charge_batiments(bounds)
st.write("Nombre de formes dans le cadastre = ", batiments.shape[0])

# Création et affichage de la carte
shapes_cadastre = []
if batiments.shape[0]>0 :
    for shp in batiments.geometry.explode():
        x,y = shp.exterior.coords.xy
        shapes_cadastre.append([x.tolist(),y.tolist()])
traces_cadastre = shape_to_traces(shapes_cadastre, nom='Cadastre', echelle=1000)

fig = px.imshow(
    orthophoto, 
    x=np.linspace(bounds.geometry[0].x/1000,bounds.geometry[1].x/1000,1024),
    y=np.linspace(bounds.geometry[0].y/1000,bounds.geometry[1].y/1000,1024),
    origin='lower',
    title="Superposition ortho/Cadastre")
fig.update_layout(
    xaxis=dict(
        dtick=0.2,
        title='X en Lambert93 / 1000',
    ),
    yaxis=dict(
        dtick=0.2,
        title='Y en Lambert93 / 1000',
    ),
)
fig.update_traces(hoverinfo='skip', hovertemplate=None)
fig.add_traces(traces_cadastre)
fig.update_layout(
    plot_bgcolor='white',
    width=1024,
    height=1024)

if page == pages[0] : 
    st.plotly_chart(fig, use_container_width=False)

model = keras.models.load_model(chemin_model)
img1 = np.array(orthophoto)[:512,:512,:]/255.0
img2 = np.array(orthophoto)[-512:,:512,:]/255.0
img3 = np.array(orthophoto)[-512:,-512:,:]/255.0
img4 = np.array(orthophoto)[:512,-512:,:]/255.0
prev_sparse = model.predict(np.array([img1,img2,img3,img4]), verbose=0)
    
# Page de modélisation
if page == pages[1] : 
    st.subheader("Choisir le modèle")

    # Prévision
    seuil=0.75
    prevision = np.array(tf.math.less(prev_sparse[:,:,:,0], 1-seuil))*1
    # Peut être qu'il est possible de faire un reshape ici :
    masque_tout=np.concatenate([prevision[0],prevision[1]], axis=0)
    masque_tout=np.concatenate([masque_tout, np.concatenate([prevision[3],prevision[2]], axis=0)], axis=1)
    
    # Contours prédiction
    ret, thresh = cv2.threshold(masque_tout.astype('uint8')*255, 125, 255, 0)
    mask_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    st.write("Nombre de bâtiments détectés par le modèle =", len(mask_contours))

    # Affichage des contours
    shapes_masque = []
    for shp in mask_contours:
        x = shp[:,0,0]*0.4
        y = shp[:,0,1]*0.4
        shapes_masque.append([x.tolist(),y.tolist()])
    traces_model = shape_to_traces(shapes_masque, nom='Modèle', 
                                   Centre=(bounds.geometry[0].x,bounds.geometry[0].y), echelle=1000,
                                   couleur='green', alpha=0.5)
    
    fig = px.imshow(
        orthophoto, 
        x=np.linspace(bounds.geometry[0].x/1000,bounds.geometry[1].x/1000,1024),
        y=np.linspace(bounds.geometry[0].y/1000,bounds.geometry[1].y/1000,1024),
        origin='lower',
        title="Superposition Ortho/Cadastre/Modèle")
    fig.update_layout(
        xaxis=dict(
            dtick=0.2,
            title='X en Lambert93 / 1000',
        ),
        yaxis=dict(
            dtick=0.2,
            title='Y en Lambert93 / 1000',
        ),
    )
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    fig.add_traces(traces_model)
    fig.add_traces(traces_cadastre)
    fig.update_layout(
        plot_bgcolor='white',
        width=1024,
        height=1024)
    st.plotly_chart(fig, use_container_width=False)

# Page de détection
if page == pages[2]: 
    c2 = st.container(border=True)
    c2.subheader("Choisir les seuils")
    seuil_certitude = 0.5 + c2.slider("Certitude (en %)",0,100,value=50,step=5)/200
    seuil_iou = c2.slider("IoU (en %)",0,100,value=25,step=5)/100
    min_pixel = round(c2.slider("Dimension minimale du bâtiment (m)",0,40,value=10,step=2)/0.4)
    
    prevision = np.array(tf.math.less(prev_sparse[:,:,:,0], 1-seuil_certitude))*1
    # Peut être qu'il est possible de faire un reshape ici :
    masque_tout=np.concatenate([prevision[0],prevision[1]], axis=0)
    masque_tout=np.concatenate([masque_tout, np.concatenate([prevision[3],prevision[2]], axis=0)], axis=1)

    # Contours prédiction
    ret, thresh = cv2.threshold(masque_tout.astype('uint8')*255, 125, 255, 0)
    mask_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    total_contours = len(mask_contours)
    st.write("Nombre de bâtiments détectés par le modèle =", total_contours)

    # Filtrage des IoU
    if batiments.shape[0]>0 :
        # st.write(type(batiments))
        raster_transform = rasterio.transform.from_bounds(
            bounds.geometry[0].x, bounds.geometry[1].y, bounds.geometry[1].x, bounds.geometry[0].y, 1024, 1024)
        masque_cadastre = rasterio.features.rasterize(batiments, transform=raster_transform, out_shape=(1024, 1024))
    else: masque_cadastre = np.zeros((1024,1024))

    shapes_filtre = []
    for shp in mask_contours:
        xmin, xmax = shp[:,0,0].min(), shp[:,0,0].max()
        ymin, ymax = shp[:,0,1].min(), shp[:,0,1].max()
        if xmax-xmin > min_pixel and ymax-ymin > min_pixel:
            # print(xmin,xmax,ymin,ymax)
            cadastre_cadre = masque_cadastre[ymin:ymax, xmin:xmax]
            prev_cadre = masque_tout[ymin:ymax, xmin:xmax]
            # st.write(calcul_IoU(cadastre_cadre, prev_cadre))
            if calcul_IoU(cadastre_cadre, prev_cadre, (min_pixel*min_pixel)//100) < seuil_iou:   
                x = shp[:,0,0]*0.4
                y = shp[:,0,1]*0.4
                shapes_filtre.append([x.tolist(),y.tolist()])
                # st.plotly_chart(px.imshow(cadastre_cadre), use_container_width=False)
                # st.plotly_chart(px.imshow(prev_cadre), use_container_width=False)
                
    traces_detect = shape_to_traces(shapes_filtre, nom='Modèle', 
                                   Centre=(bounds.geometry[0].x,bounds.geometry[0].y), echelle=1000,
                                   couleur='red', alpha=0.5)
    st.write("Nombre de bâtiments en dessous du seuil IoU =", len(traces_detect))

    # Affichage
    fig = px.imshow(
        orthophoto, 
        x=np.linspace(bounds.geometry[0].x/1000,bounds.geometry[1].x/1000,1024),
        y=np.linspace(bounds.geometry[0].y/1000,bounds.geometry[1].y/1000,1024),
        origin='lower',
        title="Superposition Ortho/Cadastre/Modèle")
    fig.update_layout(
        xaxis=dict(
            dtick=0.2,
            title='X en Lambert93 / 1000',
        ),
        yaxis=dict(
            dtick=0.2,
            title='Y en Lambert93 / 1000',
        ),
    )
    fig.update_traces(hoverinfo='skip', hovertemplate=None)
    fig.add_traces(traces_detect)
    fig.add_traces(traces_cadastre)
    fig.update_layout(
        plot_bgcolor='white',
        width=1024,
        height=1024)
    st.plotly_chart(fig, use_container_width=False)
