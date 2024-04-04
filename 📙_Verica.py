import streamlit as st
from pathlib import Path

MD_CONTENT = '/pages/content/description.md'

# page principale
# un mdenu s'affiche automatiquement à partir des fichiers .py qui sont présents dans /pages
# l'ordre du menu est l'ordre alphabétique des pages

st.set_page_config(page_title="Verification du cadastre", page_icon="📙", layout = 'wide')

st.markdown(Path(MD_CONTENT).read_text(), unsafe_allow_html=True)

