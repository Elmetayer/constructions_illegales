class assets:
	MD_CONTENT = 'pages/content/description.md'

class carte:
	CENTER_START = [48.858370, 2.294481]
	ADRESSE_DEFAUT = 'non defini'
	SIZE_DEFAUT = 200
	SIZE_MIN = 100
	SIZE_MAX = 1000
	MODE_DEFAUT = 'haut/gauche'
	MODES = [MODE_DEFAUT, 'centre']
	ZOOM_DEFAUT = 14
	EPSILON_COORD = 0.00001

class detection:
	PIXEL_SIZE_MIN = 500
	PIXEL_SIZE_MAX = 2000
	PIXEL_SIZE_DEFAULT = 1000
	PIXEL_SCALE_REF = 0.2
	SEUIL_CONF_DEFAULT = 0.05
	SEUIL_IOU_DEFAULT = 0.01
	SEUIL_AREA_DEFAULT = 10

class model_YOLO:
	YOLO_PATH = 'models/YOLOv8_20240124_bruno.pt'
	YOLO_PREDICT_CLASSES = [0]
	YOLO_SIZE = 512
	YOLO_RESOLUTION = (512, 512)

class model_Unet:
	UNET_PATH = 'models/UNet07_res512_23_12_23.h5'
	UNET_RESOLUTION = (512, 512)

class gradcam:
	OUTPUT_YOLO = ['boxes', 'conf', 'logits', 'all']
	DISPLAY_GRADCAM = ['activations', 'gradients', 'cam_heatmap']
	TARGET_LAYERS_IDX_YOLO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 18, 19, 21]
	IOU_THRESHOLD_YOLO = 0.7
	IMG_WEIGHT = 0.5
	RESOLUTION_RESULT = (256, 256)
	