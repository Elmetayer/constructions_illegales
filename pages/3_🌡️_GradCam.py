import plotly.express as px
import streamlit as st
from pages.functions.gradcam import *

# titre de la page
st.set_page_config(page_title="GradCam", page_icon="🌡️", layout = 'wide')

# variables de session
if 'fig_GradCam' not in st.session_state:
   st.session_state['fig_GradCam'] = None

OUTPUT_YOLO = ['boxes', 'conf', 'logits', 'all']
DISPLAY_GRADCAM = ['activations', 'gradients', 'cam_heatmap']
YOLO_PATH = 'models/YOLOv8_20240124_bruno.pt'
PREDICT_CLASSES = [0]
TARGET_LAYERS_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 19, 21]
SIZE = 512
RESOLUTION = (SIZE, SIZE)

##############
# Paramètres #
##############

# modèle YOLO  
@st.cache_resource
def getmodel_YOLO():
    return YOLO(YOLO_PATH)
model_YOLO = getmodel_YOLO()

# modèle YOLO GradCam
@st.cache_resource
def getmodel_YOLO_GradCam():
    return YOLO(YOLO_PATH)
model_YOLO_GradCam = getmodel_YOLO().model

conf_threshold = st.sidebar.slider('Seuil de confiance', min_value = 0.05, max_value = 0.95, value = 0.05, step = 0.05)
grads_only = st.sidebar.toggle('gradients seuls')
normalize_boxes = st.sidebar.toggle('bbox')
norm_grads_act = st.sidebar.toggle('normer grad. & act.')
abs_norm = st.sidebar.toggle('normer')
output_YOLO = st.sidebar.selectbox('sortie à analyser', OUTPUT_YOLO)
result_display = st.sidebar.selectbox('afficher', DISPLAY_GRADCAM)

#######################
# calcul et affichage #
#######################

calcul_button = st.sidebar.button('calculer')

if calcul_button:
    if st.session_state['orthophoto']:
        with st.spinner('calcul du GradCam ...'):
            @st.cache_data(show_spinner = False)
            def get_gradCam(_image):
                n_classes = len(model_YOLO.names)
                target_layers = [model_YOLO_GradCam.model[i] for i in TARGET_LAYERS_IDX]
                dict_heatmaps = make_gradCam_heatmap(_image, model_YOLO_GradCam, model_YOLO, target_layers, conf_threshold, n_classes, PREDICT_CLASSES,
                                            result_display = result_display, normalize_boxes = normalize_boxes, abs_norm = abs_norm, norm_grads_act = norm_grads_act,
                                            grads_only = norm_grads_act)
                return dict_heatmaps
            dict_heatmaps = get_gradCam(st.session_state['orthophoto'])
            @st.cache_data(show_spinner = False)
            def get_fig_gradCam(dict_heatmaps, output_YOLO):
                superposed_heatmaps = np.concatenate(
                    [np.expand_dims(cv2.resize(dict_heatmaps[output_YOLO]['layers'][layer_id]['superposed_heatmap'], RESOLUTION), 0) for layer_id in dict_heatmaps[output_YOLO]['layers'].keys()])
                fig = px.imshow(superposed_heatmaps, animation_frame=0)
                fig.update_layout(
                    height = 900,
                    width = 900)
                return fig
            st.session_state['fig_GradCam'] = get_fig_gradCam(dict_heatmaps, output_YOLO)
    else:
        st.write('⚠️ données IGN absentes')

# affichage du GradCam
if st.session_state['fig_GradCam'] is not None:
   st.plotly_chart(st.session_state['fig_GradCam'], use_container_width = True)

                                     
