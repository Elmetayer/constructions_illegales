import plotly.express as px
import streamlit as st

from pages.functions.gradcam import *
from pages.functions import config

# titre de la page
st.set_page_config(page_title="GradCam", page_icon="🌡️", layout = 'wide')

# variables de session
if 'orthophoto_GradCam' not in st.session_state:
   st.session_state['orthophoto_GradCam'] = None
if 'fig_GradCam' not in st.session_state:
   st.session_state['fig_GradCam'] = None
if 'seuil_conf_GradCam' not in st.session_state:
   if 'seuil_conf' not in st.session_state:
    st.session_state['seuil_conf_GradCam'] = config.detection.SEUIL_CONF_DEFAULT
   else:
    st.session_state['seuil_conf_GradCam'] = st.session_state['seuil_conf']

##############
# Paramètres #
##############

# modèle YOLO  
@st.cache_resource
def getmodel_YOLO():
    return YOLO(config.model_YOLO.YOLO_PATH)
model_YOLO = getmodel_YOLO()

# modèle YOLO GradCam
@st.cache_resource
def getmodel_YOLO_GradCam():
    return YOLO(config.model_YOLO.YOLO_PATH).model
model_YOLO_GradCam = getmodel_YOLO_GradCam()

conf_threshold = st.sidebar.slider('Seuil de confiance', min_value = 0.05, max_value = 0.95, value = st.session_state['seuil_conf_GradCam'], step = 0.05)
normalize_boxes = st.sidebar.toggle('bbox')
norm_grads_act = st.sidebar.toggle('normer gradients, activations')
abs_norm = st.sidebar.toggle('normer')
output_YOLO = st.sidebar.selectbox('sortie à analyser', config.gradcam.OUTPUT_YOLO)
result_display = st.sidebar.selectbox('afficher', config.gradcam.DISPLAY_GRADCAM)

#######################
# calcul et affichage #
#######################

calcul_button = st.sidebar.button('calculer')

if calcul_button:
    if 'orthophoto' in st.session_state:
        st.session_state['orthophoto_GradCam'] = st.session_state['orthophoto']
    if st.session_state['orthophoto_GradCam']:
        with st.spinner('calcul du GradCam ...'):
            @st.cache_data(show_spinner = False)
            def get_fig_gradCam(_image, _model_YOLO_GradCam, _model_YOLO, result_display, 
                                conf_threshold, normalize_boxes, abs_norm, norm_grads_act,
                output_YOLO):
                target_layers = [_model_YOLO_GradCam.model[i] for i in config.gradcam.TARGET_LAYERS_IDX_YOLO]
                dict_heatmaps = make_gradCam_heatmap(_image, model_YOLO_GradCam, model_YOLO, target_layers, conf_threshold, result_display, 
                                                     normalize_boxes = normalize_boxes, abs_norm = abs_norm, norm_grads_act = norm_grads_act)
                superposed_heatmaps = np.concatenate(
                        [np.expand_dims(cv2.resize(dict_heatmaps[output_YOLO]['layers'][layer_id]['superposed_heatmap'], config.gradcam.RESOLUTION_RESULT), 0) for layer_id in dict_heatmaps[output_YOLO]['layers'].keys()])
                fig = px.imshow(superposed_heatmaps, animation_frame = 0)
                fig.update_layout(
                    height = 900,
                    width = 900)
                return fig
            st.session_state['fig_GradCam'] = get_fig_gradCam(st.session_state['orthophoto'], model_YOLO_GradCam, model_YOLO, result_display, 
                conf_threshold, normalize_boxes, 
                abs_norm, norm_grads_act, output_YOLO)
    else:
        st.warning('⚠️ données IGN absentes')

# affichage du GradCam
if st.session_state['fig_GradCam'] is not None:
   st.plotly_chart(st.session_state['fig_GradCam'], use_container_width = True)

                                     
