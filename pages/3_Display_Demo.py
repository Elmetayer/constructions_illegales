import streamlit as st
import plotly.express as px
import rasterio
import shapely

# titre de la page
st.set_page_config(page_title="Display Demo", page_icon="👓")
st.markdown("# Display Demo")
st.sidebar.header("Display Demo")

coords_scale = 0.2

polygon_bbox = shapely.Polygon((
    (st.session_state['bbox'][0], st.session_state['bbox'][1]), 
    (st.session_state['bbox'][2], st.session_state['bbox'][1]), 
    (st.session_state['bbox'][2], st.session_state['bbox'][3]),
    (st.session_state['bbox'][0], st.session_state['bbox'][3])))
gdf_bbox = gpd.GeoDataFrame(geometry = [polygon_bbox]).set_crs(epsg = 4326)
bbox_Lambert = gdf_bbox.to_crs('EPSG:2154')
X0 = min(bbox_Lambert.geometry.x)
Y0 = max(bbox_Lambert.geometry.y)
size = max(bbox_Lambert.geometry.x) - min(bbox_Lambert.geometry.x)

raster_transform = rasterio.transform.Affine(coords_scale, 0.0, X0,
                          0.0, -coords_scale, Y0 + coords_scale*size)
xmin, ymax = raster_transform*(0, 0)
xmax, ymin = raster_transform*(size, size)
request_wms = 'https://data.geopf.fr/wms-r?LAYERS=ORTHOIMAGERY.ORTHOPHOTOS&FORMAT=image/tiff&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&STYLES=&CRS=EPSG:2154&BBOX={},{},{},{}&WIDTH={}&HEIGHT={}'.format(
   xmin, ymin, xmax, ymax, X_size, Y_size)
response_wms = requests.get(request_wms).content
orthophoto = Image.open(BytesIO(response_wms))

fig = px.imshow(orthophoto)
st.plotly_chart(fig)
