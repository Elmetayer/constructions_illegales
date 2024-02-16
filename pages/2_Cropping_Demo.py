import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import re
import rasterio
from rasterio.io import MemoryFile
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="Cropping Demo", page_icon="📈")
st.markdown("# Cropping Demo")
st.sidebar.header("Cropping Demo")

img_file = st.sidebar.file_uploader(label='Upload a file', type=['jp2', 'jpg', 'png'])
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict["1:1"]
if img_file:
    st.write(img_file.name)
    if re.findall("\.jp2$", img_file.name):
        with MemoryFile(img_file.getbuffer()) as memfile:
            with memfile.open() as dataset:
                img_array = dataset.read()
                img_array = np.transpose(img_array, [1,2,0])
                img = Image.fromarray(img_array[:1200,:1200,:])
        #with NamedTemporaryFile('wb', suffix = '.jp2', delete=False) as f:
        #    f.write(img_file.getbuffer())
        #    img = Image.open(f.name)
    else:
        img = Image.open(img_file)
    if not realtime_update:
        st.write("Double click to save crop")
    rect = st_cropper(
            img,
            realtime_update = realtime_update,
            box_color = box_color,
            aspect_ratio = aspect_ratio,
            return_type = 'box')
    raw_image = np.asarray(img).astype('uint8')
    left, top, width, height = tuple(map(int, rect.values()))
    st.write(rect)
    masked_image = np.zeros(raw_image.shape, dtype='uint8')
    masked_image[top:top + height, left:left + width] = raw_image[top:top + height, left:left + width]
    st.image(Image.fromarray(masked_image), caption='masked image')
    
    st.write("Preview")
    _ = Image.fromarray(masked_image).thumbnail((150,150))
    st.image(Image.fromarray(masked_image))