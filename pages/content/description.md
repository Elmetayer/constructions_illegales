# Outil de vérification du cadastre

L'outil met en oeuvre un modèle de deep learning pour reconnaître des bâtiments à partir d'une image satellite. Les bâtiments détectés sont ensuite comparés aux données cadastrales afin de détecter les différences.

## Modèles

Plusieurs modèles peuvent être utilisés :

**YOLO**

description : modèle YOLOv8
taille : 512x512
paramètres : 27 285 968 

> référence : [ultralytics](https://docs.ultralytics.com/models/yolov8/)

**Unet**

description : modèle de type encoder-decoder (en "U")
taille : 512x512
paramètres : 1 941 979

> référence : [Keras](https://keras.io/examples/vision/oxford_pets_image_segmentation/)

### GradCam

La technique "GradCam", pour "Gradient-weighted Class Activation Mapping" s’appuie sur le calcul du gradient d’une valeur de perte ("loss") pour une classe c donnée par rapport au résultat de la fonction d’activation en sortie d’un layer donné (typiquement, un layer de convolution).

**Le calcul n'est disponible que pour le modèle YOLO** *(en cours pour le modèle Unet ...)*

> références utilisées :
>
> + package [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
> + package [YOLO-V8-CAM](https://github.com/rigvedrs/YOLO-V8-CAM)
> + [dépôt](https://github.com/z1069614715/objectdetection_script/blob/master/yolo-gradcam/yolov8_heatmap.py) avec un exemple qui semble fonctionner

## Données utilisées

### Entraînement des modèles

Les modèles ont été entraînés sur les données BDTOPO ET BDORTHO de l'IGN du **département du Rhône (69)**

**Images satellites (BDORTHO)**

[Images géographiques](https://geoservices.ign.fr/bdortho) du territoire national (France vue du ciel) au format jp2. Les images sont des dalles carrées jointives, sans recouvrement. Les résolutions nominales du pixel sont les suivantes :

+ départements d’Île-de-France : 15cm
+ tous les départements (dont Île-de-France) et collectivités d’outre-mer : 20cm

Les coordonnées sont exprimées en Lambert-93 pour la France métropolitaine.

**Infrastructures (BDTOPO)**

[Modélisation 2D et 3D]() du territoire et de ses infrastructures sur l'ensemble du territoire français. Les objets au format Shapefile, et regroupés en 8 thèmes (cf. modélisation INSPIRE, “Infrastructure for Spatial Information in Europe”) : 

+ administratif : limites et unités administratives 
+ bâti : constructions 
+ hydrographie : éléments ayant trait à l’eau 
+ lieux nommés : lieu ou lieu-dit possédant un toponyme et décrivant un espace naturel ou un lieu habité 
+ occupation du sol : végétation, estran 
+ services et activités : services publics, stockage et transport des sources d'énergie, lieux et sites industriels 
+ transport : infrastructures du réseau routier, ferré et aérien 
+ zones réglementées : zonages faisant l'objet de réglementations spécifique

### Outil

Dans l'outil, les modèles sont appliquées aux données BDORTHO (voir-ci-dessus), puis comparés aux données cadastrales.

Les données sont récupérées depuis les API IGN (voir [documentation](https://geoservices.ign.fr/documentation/services/services-geoplateforme))

**Images satellites (BDORTHO)**

> Exemple de requête :
> https://data.geopf.fr/wms-r?LAYERS=HR.ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}

**Cadastre**

Données cadastrales compilées par [Etalab](https://geoservices.ign.fr/bdortho) et décomposées en 8 catégories :

+ parcelles
+ subdivisions_fiscales
+ lieux_dits
+ feuilles
+ sections
+ prefixes_sections
+ communes
+ batiments (code 01 pour un bâtiment dur et code 02 pour un bâtiment léger)

Seuls les objets de type "bâtiment" sont récupérés.

> Exemple de requête :
> https://data.geopf.fr/wfs?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&typename=CADASTRALPARCELS.PARCELLAIRE_EXPRESS:batiment&outputformat=application/json&BBOX={},{},{},{} 

## Fonctionnement de l'outil


