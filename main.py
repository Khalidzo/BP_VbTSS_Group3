import cv2 as cv
import json
import pickle
from speed_detection.config import PIXELS_PER_METER
from ultralytics import YOLO
from speed_detection.byte_tracker import BYTETracker, Track
from speed_detection.roi_selector import ROISelector
from speed_detection.visualization import (
    draw_ground_truth_speeds,
    draw_rois,
    draw_tracks,
)
from speed_detection.roi_processing import (
    process_rois_for_frame,
    cleanup_old_vehicle_data,
)
from speed_detection.utils import format_detections

session = "session6_left"

# ──────── Paths & Parameters ────────
video_path = rf"C:\\Users\\yongue-tchanga\\Desktop\\Bp\\BP_VbTSS_Group3\\traffic_analysis\\test.mp4"

# ──────── Initialization ────────
cap = cv.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)
ret, first_frame = cap.read()
if not ret:
    raise IOError("Could not read first frame")

# ──────── Select Number of ROIs ────────
N_ROIS = int(input("\nHow many ROIs do you want to select? (Enter an integer)\n> "))

# ──────── ROI Selection ────────
roi_selector = ROISelector(first_frame, N_ROIS)
roi_data = roi_selector.select_rois()

if roi_data is None:
    print("ROI selection cancelled by user")
    cap.release()
    exit()

# Calculate perspective transforms
roi_selector.calculate_perspective_transforms(PIXELS_PER_METER)

# ByteTrack Tracker
tracker = BYTETracker(track_thresh=0.3, track_buffer=30, match_thresh=0.7)

# YOLO model
model = YOLO("yolo11n.pt")
model.to("cuda")

# Dictionary to store vehicle tracking data with Kalman filters for each ROI
vehicle_data = {}  # {roi_id: {track_id: {'kalman_filter': KalmanSpeedFilter, ...}}}

for roi in roi_data:
    vehicle_data[roi["id"]] = {}


# ──────── Load Ground Truth Data ────────
# Load JSON tracking
with open(
    rf"D:\\2016-ITS-BrnoCompSpeed\\results\\{session}\\system_dubska_bmvc14.json", "r"
) as f:
    json_data = json.load(f)

# Load GT speed data
with open(rf"D:\\2016-ITS-BrnoCompSpeed\\dataset\\{session}\\gt_data.pkl", "rb") as f:
    gt_data = pickle.load(f, encoding="latin1")

# Build carId ➝ speed map
gt_speeds = {car["carId"]: car["speed"] for car in gt_data["cars"]}

# Attach real speed to each tracked car
for car in json_data["cars"]:
    car["real_speed"] = gt_speeds.get(car["id"])

# ──────── Video Frame Playback ────────
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / VIDEO_FPS
    dt = 1.0 / VIDEO_FPS

    # Draw Ground Truth speeds
    draw_ground_truth_speeds(frame, frame_count, json_data["cars"])

    # YOLO detection
    results = model(frame, verbose=False)[0]

    # Format detections
    bboxes, scores, class_ids = format_detections(results)

    # Apply ROI overlays
    frame = draw_rois(frame, roi_data)

    # Update tracker
    byte_tracks = tracker.update(bboxes, scores, class_ids)
    tracks = [Track(t) for t in byte_tracks]

    # Process all ROIs and tracks
    process_rois_for_frame(tracks, roi_data, vehicle_data, current_time, dt)

    # Draw tracks and speed annotations
    draw_tracks(frame, tracks, roi_data, vehicle_data)

    # Clean up old data
    cleanup_old_vehicle_data(vehicle_data, tracks, current_time)

    # Show final frame
    cv.imshow("Multi-ROI Vehicle Speed Tracking", cv.resize(frame, (800, 600)))
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
