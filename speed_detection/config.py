import cv2 as cv

# Real-world parameters
LANE_LEN_M = 2
GAP_LEN_M = 2.5
LANE_WIDTH_M = 3
PIXELS_PER_METER = 30

# YOLO configuration
YOLO_THRESHOLD = 0.3
YOLO_CLASSES = [2, 3, 5, 7]  # e.g., car, motorcycle, bus, truck

# Colors and font
FONT = cv.FONT_HERSHEY_SIMPLEX
COL_RED, COL_GRN, COL_BLU = (0, 0, 255), (0, 255, 0), (255, 0, 0)
COL_GREY, COL_HELPER = (50, 50, 50), (180, 180, 180)

ROI_COLORS = [
    (144, 238, 144),
    (255, 182, 193),
    (173, 216, 230),
    (255, 218, 185),
    (221, 160, 221),
    (255, 255, 224),
    (255, 160, 122),
    (176, 196, 222),
]
