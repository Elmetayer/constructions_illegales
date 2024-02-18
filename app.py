import streamlit as st

# page principale
# un mdenu s'affiche automatiquement à partir des fichiers .py qui sont présents dans /pages
# l'ordre du menu est l'ordre alphabétique des pages

st.set_page_config(
    page_title="Démo",
    page_icon="👋",
)
st.header("Démo")
st.markdown(
    """
    Quelques démos
    """
    )
st.sidebar.success("Choisir une démo")

