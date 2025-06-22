# tracking.py
import cv2 as cv
import numpy as np
from kalman_filter import KalmanSpeedFilter
from utils import point_in_roi
from config import PIXELS_PER_METER


# Function to convert pixel position to real-world coordinates for a specific ROI
def pixel_to_meter(pixel_point, roi):
    px, py = pixel_point
    point = np.array([[px, py]], dtype=np.float32)
    transformed = cv.perspectiveTransform(point.reshape(-1, 1, 2), roi["M"]).reshape(
        -1, 2
    )[0]
    meter_x = transformed[0] / PIXELS_PER_METER
    meter_y = transformed[1] / PIXELS_PER_METER
    return meter_x, meter_y


# Function to calculate traditional speed (for comparison)
def calculate_traditional_speed(positions, timestamps):
    if len(positions) < 2:
        return None

    start_pos, end_pos = positions[0], positions[-1]
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = np.sqrt(dx**2 + dy**2)

    time_elapsed = timestamps[-1] - timestamps[0]
    if time_elapsed <= 0:
        return None

    speed_ms = distance / time_elapsed
    speed_kmh = speed_ms * 3.6

    return speed_kmh


def process_rois_for_frame(tracks, roi_data, vehicle_data, current_time, dt):
    for roi in roi_data:
        if not roi["completed"]:
            continue

        roi_id = roi["id"]
        roi_vehicle_data = vehicle_data[roi_id]

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            bottom_center = (int((l + r) / 2), b)

            in_roi = point_in_roi(bottom_center, roi["roi_pts"])

            if track_id not in roi_vehicle_data:
                roi_vehicle_data[track_id] = {
                    "kalman_filter": KalmanSpeedFilter(dt=dt),
                    "positions": [],
                    "timestamps": [],
                    "in_roi": False,
                    "kalman_speed": None,
                    "traditional_speed": None,
                    "position_history": [],
                }

            track_data = roi_vehicle_data[track_id]
            kalman_filter = track_data["kalman_filter"]

            if in_roi:
                real_pos = pixel_to_meter(bottom_center, roi)
                kalman_filter.predict()
                kalman_filter.update(real_pos)

                track_data["positions"].append(real_pos)
                track_data["timestamps"].append(current_time)
                track_data["position_history"].append(bottom_center)
                track_data["in_roi"] = True
                track_data["kalman_speed"] = kalman_filter.get_speed_kmh()

                if len(track_data["positions"]) >= 2:
                    track_data["traditional_speed"] = calculate_traditional_speed(
                        track_data["positions"], track_data["timestamps"]
                    )

            elif track_data["in_roi"]:
                kalman_filter.predict()
                track_data["kalman_speed"] = kalman_filter.get_speed_kmh()

                if len(track_data["positions"]) >= 2:
                    track_data["traditional_speed"] = calculate_traditional_speed(
                        track_data["positions"], track_data["timestamps"]
                    )
                track_data["in_roi"] = False
            else:
                kalman_filter.predict()


def cleanup_old_vehicle_data(vehicle_data, tracks, current_time, stale_threshold=2.0):
    current_track_ids = {t.track_id for t in tracks if t.is_confirmed()}

    for roi_id in vehicle_data:
        old_track_ids = set(vehicle_data[roi_id].keys()) - current_track_ids
        for track_id in old_track_ids:
            timestamps = vehicle_data[roi_id][track_id]["timestamps"]
            if timestamps and (current_time - timestamps[-1]) > stale_threshold:
                vehicle_data[roi_id].pop(track_id, None)
