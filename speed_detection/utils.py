import cv2 as cv
from speed_detection.config import YOLO_THRESHOLD, YOLO_CLASSES


def draw_label_with_bg(
    img,
    text,
    topleft,
    font_scale: float = 1.0,
    text_color=(255, 255, 255),
    bg_color=(255, 100, 100),
):
    font = cv.FONT_HERSHEY_SIMPLEX
    ((w, h), _) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)
    x, y = topleft
    cv.rectangle(img, (x, y - h - 4), (x + w, y + 2), bg_color, -1)
    cv.putText(
        img,
        text,
        (x, y - 2),
        font,
        font_scale,
        text_color,
        thickness=1,
        lineType=cv.LINE_AA,
    )


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
