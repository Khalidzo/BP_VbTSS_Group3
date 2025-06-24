import cv2 as cv
import torch
from ultralytics import YOLO
from InquirerPy import inquirer
from config import TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT
from speed_detection.speed_estimator import SpeedEstimator
from traffic_jam_detection.detect_congestion import CongestionDetector

# ──────── Video Setup ────────
session = "session6_left"
video_path = rf"D:\\2016-ITS-BrnoCompSpeed\\dataset\\{session}\\video.avi"

cap = cv.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)
ret, first_frame = cap.read()
if not ret:
    raise IOError("Could not read first frame")
resized_first_frame = cv.resize(first_frame, (TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT))

# ──────── Feature Selection (Interactive) ────────
feature_choices = [
    {"name": "Speed Detection", "value": "speed"},
    {"name": "Traffic Jam Detection", "value": "congestion"},
]

selected_feature_keys = inquirer.checkbox( # type: ignore
    message="Select features to activate (use space to select):",
    choices=feature_choices,
    instruction="(Use arrow keys to navigate, space to select, enter to confirm)",
).execute()

# ──────── Feature Initialization ────────
selected_features = []

if "speed" in selected_feature_keys:
    speed_estimation = SpeedEstimator()
    speed_estimation.get_user_input(resized_first_frame)
    selected_features.append(speed_estimation)

if "congestion" in selected_feature_keys:
    congestion_detection = CongestionDetector(video_fps=VIDEO_FPS)
    congestion_detection.get_user_input(resized_first_frame)
    selected_features.append(congestion_detection)

if not selected_features:
    print("❌ No features selected. Exiting.")
    exit()

# ──────── Shared YOLO Model ────────
model = YOLO("yolo11n.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ──────── Main Loop ────────
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv.resize(frame, (TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT))
    current_time = frame_count / VIDEO_FPS
    dt = 1.0 / VIDEO_FPS

    detections = model(frame, verbose=False)[0]

    processed_frame = frame.copy()
    for feature in selected_features:
        processed_frame = feature.process_frame(
            processed_frame, current_time, dt, detections, model.names
        )

    cv.imshow("Vehicle Feature Processor", processed_frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

# ──────── Cleanup ────────
cap.release()
cv.destroyAllWindows()

# Optional: Finalize and print results
for feature in selected_features:
    if hasattr(feature, "finalize"):
        feature.finalize()
    if hasattr(feature, "print_results"):
        feature.print_results()
