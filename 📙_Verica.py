import streamlit as st
from pathlib import Path

from pages.functions import config

# page principale
# un mdenu s'affiche automatiquement à partir des fichiers .py qui sont présents dans /pages
# l'ordre du menu est l'ordre alphabétique des pages

st.set_page_config(page_title="Verification du cadastre", page_icon="📙", layout = 'wide')

st.markdown(Path(config.assets.MD_CONTENT).read_text(), unsafe_allow_html=True)

