import cv2 as cv
from speed_detection.config import YOLO_THRESHOLD, YOLO_CLASSES

# Function to check if a point is inside the ROI polygon
def point_in_roi(point, roi_polygon):
    return cv.pointPolygonTest(roi_polygon, point, False) >= 0


def format_detections(results):
    bboxes, scores, class_ids = [], [], []

    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])

        if conf > YOLO_THRESHOLD and cls in YOLO_CLASSES:
            bboxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to x, y, w, h
            scores.append(conf)
            class_ids.append(cls)

    return bboxes, scores, class_ids
