import streamlit as st

# Upload an image and set some options for demo purposes

st.set_page_config(
    page_title="Démo",
    page_icon="👋",
)
st.header("VeriCa")
st.sidebar.success("Choisir une démo")

st.markdown(
    """
    Quelques démos
    """
    )
