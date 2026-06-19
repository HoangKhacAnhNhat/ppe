YOLOV8_PATH = 'checkpoint\yolov8m_mix-aug.pt'
YOLOV5_PATH = 'checkpoint\yolov5mu_mix-aug.pt'
RTDETR_PATH = r'checkpoint\rtdetrl_mix-aug.pt'

VIDEO_PATH = 'videos\YTDown_Shorts_construction-tap-trung-tbm-dau-gio-o-du-_Media_0dlE0Gv8ONE_001_1080p.mp4'

FRAME_DIR = "logs/frames"
LOG_PATH = "logs/ppe_event_log.csv"

model_dict = {
    'rtdetr': RTDETR_PATH,
    'yolov8': YOLOV8_PATH,
    'yolov5': YOLOV5_PATH
}

mode_dict = {
    'Camera': 0,
    'Video': VIDEO_PATH
}

VIOLATION_CLASSES = [
    "no_helmet",
    "no_vest",
    "no_boots",
    "no_gloves"
]