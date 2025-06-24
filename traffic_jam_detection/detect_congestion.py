import cv2
from traffic_jam_detector import TrafficJamDetector
from config import Config

# Main execution following code 1 structure
session = "session6_left"
video_path = rf"D:\\2016-ITS-BrnoCompSpeed\\dataset\\{session}\\video.avi"

# ──────── Initialization ────────
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video file '{video_path}'")

VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
ret, first_frame = cap.read()
if not ret:
    raise IOError("Could not read first frame")

# Resize first frame
first_frame = cv2.resize(first_frame, (Config.TARGET_WIDTH, Config.TARGET_HEIGHT))

# ──────── ROI Selection ────────
detector = TrafficJamDetector()
detector._select_rois(first_frame)

if not detector.polygons:
    print("No ROIs selected, exiting")
    cap.release()
    exit()

# ──────── Video Frame Playback ────────
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / VIDEO_FPS

    # Resize frame to target dimensions
    frame = cv2.resize(frame, (Config.TARGET_WIDTH, Config.TARGET_HEIGHT))

    # Process detections and tracking
    tracked_objects = detector._process_detections_and_tracking(frame)

    # Analyze ROIs
    zone_vehicle_counts, zone_velocities_this_frame, zone_stationary_vehicles = (
        detector._process_roi_analysis(tracked_objects, frame_count)
    )

    # Update jam detection
    detector._update_jam_detection(
        zone_vehicle_counts,
        zone_velocities_this_frame,
        zone_stationary_vehicles,
        frame_count,
    )

    # Draw visualization
    detector._draw_visualization(frame, tracked_objects)

    # Show frame
    cv2.imshow("Traffic Jam Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# ──────── Cleanup and Results ────────
# Close open jams at the end
last_frame_idx = frame_count
for idx, active in enumerate(detector.zone_active_jams):
    if active:
        start_frame = detector.jam_start_frames[idx]
        if start_frame != -1:
            detector.jam_intervals[idx].append((start_frame, last_frame_idx))

cap.release()
cv2.destroyAllWindows()

# Generate results and evaluation (similar to code 1's final processing)
print("\n--- Traffic Jam Detection Results ---")
for zone_idx, intervals in enumerate(detector.jam_intervals):
    print(f"Zone {zone_idx + 1}:")
    if not intervals:
        print("  No jams detected.")
    else:
        for start_f, end_f in intervals:
            start_s = round(start_f / VIDEO_FPS, 2)
            end_s = round(end_f / VIDEO_FPS, 2)
            print(f"  Jam from {start_s}s to {end_s}s (Frames: {start_f}-{end_f})")

print("\n--- Traffic Jam Detection Finished ---")
