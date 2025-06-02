import cv2 as cv
import numpy as np
import json
import pickle
from ultralytics import YOLO
from byte_tracker import BYTETracker, Track
from kalman_filter import KalmanSpeedFilter

session = "session6_left"

# ──────── Paths & Parameters ────────
video_path = fr"D:\\2016-ITS-BrnoCompSpeed\\dataset\\{session}\\video.avi"
LANE_LEN_M = 2
GAP_LEN_M = 2.5
LANE_WIDTH_M = 3

# ──────── Initialization ────────
cap = cv.VideoCapture(video_path)
VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)
ret, first_frame = cap.read()
if not ret:
    raise IOError("Could not read first frame")

# ──────── Select Stripes & Lanes ────────
N_STRIPES = int(input("\nHow many stripes are there initially? (Enter an integer; stripes must be consecutive)\n> "))
N_LANES = int(input("\nHow many lanes are there? (Enter an integer; lanes must be consecutive)\n> "))

# ByteTrack Tracker
tracker = BYTETracker(track_thresh=0.3, track_buffer=30, match_thresh=0.7)

# YOLO model
model = YOLO("yolo11n.pt")
model.to('cuda')

points, endpoints, lane_pts = [], [], []
cursor_pos = None
mode = "STRIPES"
banner_h = 40

# Colors & fonts
FONT = cv.FONT_HERSHEY_SIMPLEX
COL_RED, COL_GRN, COL_BLU = (0,0,255), (0,255,0), (255,0,0)
COL_GREY, COL_HELPER = (50,50,50), (180,180,180)

# Calculate scale factors for resizing
scale_factor_x = 800 / first_frame.shape[1]
scale_factor_y = 600 / first_frame.shape[0]

# Functions to convert between original and resized coordinates
def screen_to_original(px, py):
    """Convert screen coordinates to original frame coordinates"""
    return int(px / scale_factor_x), int(py / scale_factor_y)

def original_to_screen(px, py):
    """Convert original frame coordinates to screen coordinates"""
    return int(px * scale_factor_x), int(py * scale_factor_y)

# Banner text
def banner_text():
    if mode == "STRIPES":
        return "Click the START and END points of the stripe sequence"
    if mode == "ENDPOINTS":
        return f"Click Road {'Start' if len(endpoints)==0 else 'End'} ({len(endpoints)}/2)"
    if mode == "LANE":
        return f"Click rectangle corners TL,TR,BR,BL ({len(lane_pts)}/4)"
    return "SPACE starts video  |  ESC exits"

# Mouse handler
def on_mouse(event, x, y, flags, param):
    global cursor_pos, mode
    
    # Convert screen coordinates to original frame coordinates
    orig_x, orig_y = screen_to_original(x, y)
    cursor_pos = (orig_x, orig_y)

    if y < banner_h or event != cv.EVENT_LBUTTONDOWN:
        return

    if mode == "STRIPES" and len(points) < 2:
        points.append((orig_x, orig_y))
    elif mode == "ENDPOINTS" and len(endpoints) < 2:
        endpoints.append((orig_x, orig_y))
    elif mode == "LANE" and len(lane_pts) < 4:
        lane_pts.append((orig_x, orig_y))
        if len(lane_pts) == 4:
            mode = "DONE"

cv.namedWindow("Select")
cv.setMouseCallback("Select", on_mouse)
sel_img = first_frame.copy()

# UI Loop
while mode != "DONE":
    # Create a resized version for display
    sel_img_resized = cv.resize(sel_img, (800, 600))
    vis = sel_img_resized.copy()
    
    # Add banner
    cv.rectangle(vis, (0, 0), (vis.shape[1], banner_h), COL_GREY, -1)
    cv.putText(vis, banner_text(), (10, 28), FONT, 0.7, (255, 255, 255), 2, cv.LINE_AA)

    # Draw points on resized image
    if mode == "STRIPES":
        for i, (px, py) in enumerate(points):
            # Convert original coordinates to screen coordinates
            sx, sy = original_to_screen(px, py)
            cv.circle(vis, (sx, sy), 4, COL_RED, -1)
            cv.putText(vis, f"{i+1}", (sx+6, sy-6), FONT, 0.5, COL_RED, 1)
            
    for i, (sx, sy) in enumerate(endpoints):
        # Convert to screen coordinates
        disp_x, disp_y = original_to_screen(sx, sy)
        cv.circle(vis, (disp_x, disp_y), 5, COL_GRN, -1)
        cv.putText(vis, "Start" if i == 0 else "End", (disp_x+6, disp_y-6), FONT, 0.6, COL_GRN, 1)
        
    for i, (rx, ry) in enumerate(lane_pts):
        # Convert to screen coordinates
        disp_x, disp_y = original_to_screen(rx, ry)
        cv.circle(vis, (disp_x, disp_y), 5, (200,255,200), -1)
        cv.putText(vis, str(i+1), (disp_x+6, disp_y-6), FONT, 0.6, (200,255,200), 1)

    if mode == "LANE" and len(endpoints) == 2:
        end_pt = np.array(endpoints[1])
        if len(lane_pts) == 0 and cursor_pos is not None:
            cur = np.array(cursor_pos)
            dv = end_pt - cur
            if np.linalg.norm(dv) > 0:
                dv = dv / np.linalg.norm(dv)
                # Convert line endpoints to screen coordinates
                start_point = original_to_screen(*(cur - 2000*dv).astype(int))
                end_point = original_to_screen(*(cur + 2000*dv).astype(int))
                cv.line(vis, start_point, end_point, COL_HELPER, 1, cv.LINE_AA)

        if len(lane_pts) == 1:
            p0 = np.array(lane_pts[0])
            dv = end_pt - p0
            dv = dv / np.linalg.norm(dv)
            # Convert line endpoints to screen coordinates
            start_point = original_to_screen(*(p0 - 2000*dv).astype(int))
            end_point = original_to_screen(*(p0 + 2000*dv).astype(int))
            cv.line(vis, start_point, end_point, COL_HELPER, 1, cv.LINE_AA)

        if len(lane_pts) == 2 or len(lane_pts) == 3:
            tl, tr = np.array(lane_pts[0]), np.array(lane_pts[1])
            start_pt = np.array(endpoints[0])
            ed = tr - tl
            ed = ed / np.linalg.norm(ed)
            # Convert line endpoints to screen coordinates
            start_point = original_to_screen(*(start_pt - 2000*ed).astype(int))
            end_point = original_to_screen(*(start_pt + 2000*ed).astype(int))
            cv.line(vis, start_point, end_point, COL_HELPER, 1, cv.LINE_AA)

    cv.imshow("Select", vis)
    k = cv.waitKey(20) & 0xFF

    if k == 27:
        cap.release(); cv.destroyAllWindows(); exit()

    if k == ord(' '):
        if mode == "STRIPES" and len(points) == 2:
            # Interpolate the stripes from start to end point
            start_pt, end_pt = np.array(points[0]), np.array(points[1])
            stripe_vec = end_pt - start_pt
            stripe_vec = stripe_vec / np.linalg.norm(stripe_vec)
            stripe_total_len = N_STRIPES * LANE_LEN_M + (N_STRIPES - 1) * GAP_LEN_M
            unit_px = np.linalg.norm(end_pt - start_pt) / stripe_total_len
            stripe_px_len = LANE_LEN_M * unit_px
            gap_px_len = GAP_LEN_M * unit_px

            generated_points = []
            cursor = np.array(start_pt, dtype=np.float32)
            for _ in range(N_STRIPES):
                p1 = cursor.copy()
                p2 = p1 + stripe_vec * stripe_px_len
                generated_points.append(tuple(p1.astype(int)))
                generated_points.append(tuple(p2.astype(int)))
                cursor = p2 + stripe_vec * gap_px_len

            points = generated_points
            endpoints = [points[0], points[-1]]
            mode = "LANE"

        elif mode == "ENDPOINTS" and len(endpoints) == 2:
            mode = "LANE"

cv.destroyWindow("Select")


def draw_label_with_bg(img, text, topleft, font_scale: float = 1.0, text_color=(255,255,255), bg_color=(255, 100, 100)):
    font = cv.FONT_HERSHEY_SIMPLEX
    ((w, h), _) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)
    x, y = topleft
    cv.rectangle(img, (x, y - h - 4), (x + w, y + 2), bg_color, -1)
    cv.putText(img, text, (x, y - 2), font, font_scale, text_color, thickness=1, lineType=cv.LINE_AA)

# ──────── Calculate Perspective Transform Matrix ────────
# Sort the ROI points to ensure they are in order: TL, TR, BR, BL
pts = np.array(lane_pts, np.float32)
roi_pts = pts[np.argsort(np.arctan2(pts[:,1]-pts.mean(0)[1], pts[:,0]-pts.mean(0)[0]))].astype(np.int32)

# Define real-world dimensions
real_roi_length = N_STRIPES * LANE_LEN_M + (N_STRIPES - 1) * GAP_LEN_M  # Length in meters
real_roi_width = N_LANES * LANE_WIDTH_M  # Width in meters

# Define the destination points for the bird's eye view (in pixels per meter)
PIXELS_PER_METER = 30  # Scale factor for visualization
dst_width = int(real_roi_width * PIXELS_PER_METER)
dst_length = int(real_roi_length * PIXELS_PER_METER)

# Define destination points (bird's eye view)
dst_pts = np.array([
    [0, 0],                    # Top left
    [dst_width, 0],            # Top right
    [dst_width, dst_length],   # Bottom right
    [0, dst_length]            # Bottom left
], dtype=np.float32)

# Get the perspective transform matrix
M = cv.getPerspectiveTransform(roi_pts.astype(np.float32), dst_pts)
inv_M = cv.getPerspectiveTransform(dst_pts, roi_pts.astype(np.float32))

# Dictionary to store vehicle tracking data with Kalman filters
vehicle_data = {}  # {track_id: {'kalman_filter': KalmanSpeedFilter, 'positions': [], 'timestamps': [], 'in_roi': False, 'speed': None}}

# Function to check if a point is inside the ROI polygon
def point_in_roi(point, roi_polygon):
    return cv.pointPolygonTest(roi_polygon, point, False) >= 0

# Function to convert pixel position to real-world coordinates (meters)
def pixel_to_meter(pixel_point):
    # Apply perspective transform to get bird's eye view coordinates
    px, py = pixel_point
    point = np.array([[px, py]], dtype=np.float32)
    transformed = cv.perspectiveTransform(point.reshape(-1, 1, 2), M).reshape(-1, 2)[0]
    # Convert from bird's eye view pixels to meters
    meter_x = transformed[0] / PIXELS_PER_METER
    meter_y = transformed[1] / PIXELS_PER_METER
    return meter_x, meter_y

# Function to calculate traditional speed (for comparison)
def calculate_traditional_speed(positions, timestamps):
    if len(positions) < 2:
        return None
    
    # Calculate distance between first and last position (meters)
    start_pos, end_pos = positions[0], positions[-1]
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance = np.sqrt(dx**2 + dy**2)  # meters
    
    # Calculate time elapsed (seconds)
    time_elapsed = timestamps[-1] - timestamps[0]
    if time_elapsed <= 0:
        return None
    
    # Calculate speed (m/s)
    speed_ms = distance / time_elapsed
    
    # Convert to km/h
    speed_kmh = speed_ms * 3.6
    
    return speed_kmh

# ──────── Video Frame Playback ────────
cap.set(cv.CAP_PROP_POS_FRAMES, 0)
frame_count = 0
prev_frame_time = frame_count / VIDEO_FPS

# Load JSON tracking
with open(fr"D:\\2016-ITS-BrnoCompSpeed\\results\\{session}\\system_dubska_bmvc14.json", "r") as f:
    json_data = json.load(f)

# Load GT speed data
with open(fr"D:\\2016-ITS-BrnoCompSpeed\\dataset\\{session}\\gt_data.pkl", "rb") as f:
    gt_data = pickle.load(f, encoding="latin1")

# Build carId ➝ speed map
gt_speeds = {car['carId']: car['speed'] for car in gt_data['cars']}

# Attach real speed to each tracked car (by shared id)
for car in json_data['cars']:
    car['real_speed'] = gt_speeds.get(car['id'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Calculate time step for Kalman filter
    current_time = frame_count / VIDEO_FPS
    dt = 1.0 / VIDEO_FPS

    # Draw Ground Truth speed for all cars
    for car in json_data['cars']:
        if frame_count in car['frames']:
            i = car['frames'].index(frame_count)
            # Get original positions
            orig_x, orig_y = int(car['posX'][i]), int(car['posY'][i])
            
            # Convert to screen coordinates for display
            disp_x, disp_y = original_to_screen(orig_x, orig_y)

            # Draw real speed
            if car.get('real_speed') is not None:
                speed_text = f"GT: {car['real_speed']:.1f} km/h"
                draw_label_with_bg(frame, speed_text, (orig_x, orig_y - 25), 
                                  font_scale=0.6, bg_color=(0, 100, 255))
                
                # Draw position marker in original coordinates
                cv.circle(frame, (orig_x, orig_y), 5, (0, 100, 255), -1)

    # YOLO detection
    results = model(frame, verbose=False)[0]
    
    # Format detections for ByteTrack
    bboxes, scores, class_ids = [], [], []
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        if conf > 0.3 and cls in [2, 3, 5, 7]:  # car classes
            bboxes.append([x1, y1, x2-x1, y2-y1])  # convert to [x,y,w,h] format for tracker
            scores.append(conf)
            class_ids.append(cls)
    
    # Draw ROI overlay
    overlay = frame.copy()
    cv.fillPoly(overlay, [roi_pts], (144, 238, 144))
    frame = cv.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Draw middle roi line
    top_mid = tuple(((roi_pts[0] + roi_pts[1]) // 2))
    bottom_mid = tuple(((roi_pts[2] + roi_pts[3]) // 2))
    cv.line(frame, top_mid, bottom_mid, COL_GRN, 2)

    # Update ByteTrack tracker and get tracks
    byte_tracks = tracker.update(bboxes, scores, class_ids)
    
    # Convert ByteTrack outputs to Track objects for compatibility
    tracks = [Track(t) for t in byte_tracks]

    # Draw tracked boxes and IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        
        # Calculate the bottom center point of the bounding box
        bottom_center = (int((l + r) / 2), b)
        
        # Check if the bottom center is inside the ROI
        in_roi = point_in_roi(bottom_center, roi_pts)
        
        # Initialize track data if this is a new track
        if track_id not in vehicle_data:
            vehicle_data[track_id] = {
                'kalman_filter': KalmanSpeedFilter(dt=dt),
                'positions': [],
                'timestamps': [],
                'in_roi': False,
                'kalman_speed': None,
                'traditional_speed': None,
                'position_history': []  # Store pixel positions for visualization
            }
        
        # Update track data
        track_data = vehicle_data[track_id]
        kalman_filter = track_data['kalman_filter']
        
        # If bottom center is in ROI, update position and timestamp
        if in_roi:
            # Convert pixel position to real-world position (meters)
            real_pos = pixel_to_meter(bottom_center)
            
            # Update Kalman filter with new measurement
            kalman_filter.predict()
            kalman_filter.update(real_pos)
            
            # Save position and timestamp for traditional method
            track_data['positions'].append(real_pos)
            track_data['timestamps'].append(current_time)
            track_data['position_history'].append(bottom_center)
            
            # Set in_roi flag if this is the first time in ROI
            if not track_data['in_roi']:
                track_data['in_roi'] = True
            
            # Get Kalman filtered speed
            track_data['kalman_speed'] = kalman_filter.get_speed_kmh()
            
            # Calculate traditional speed for comparison
            if len(track_data['positions']) >= 2:
                track_data['traditional_speed'] = calculate_traditional_speed(
                    track_data['positions'], track_data['timestamps'])
                
        elif track_data['in_roi']:
            # If the vehicle was in ROI but now left, just predict without update
            kalman_filter.predict()
            track_data['kalman_speed'] = kalman_filter.get_speed_kmh()
            
            # Finalize traditional speed calculation
            if len(track_data['positions']) >= 2:
                track_data['traditional_speed'] = calculate_traditional_speed(
                    track_data['positions'], track_data['timestamps'])
            track_data['in_roi'] = False
        else:
            # Vehicle not in ROI, just predict to maintain filter state
            kalman_filter.predict()
        
        # Draw the bounding box
        cv.rectangle(frame, (l, t), (r, b), COL_BLU, 2)
        
        # Display track ID
        draw_label_with_bg(frame, f"ID {track_id}", (l, t))
        
        # Display Kalman filtered speed
        if track_data['kalman_speed'] is not None and track_data['kalman_speed'] > 0:
            speed_text = f"Kalman: {track_data['kalman_speed']:.1f} km/h"
            draw_label_with_bg(frame, speed_text, (l, t-25), bg_color=(50, 200, 50))
        
        # Display traditional speed for comparison (optional)
        if track_data['traditional_speed'] is not None and track_data['traditional_speed'] > 0:
            trad_speed_text = f"Trad: {track_data['traditional_speed']:.1f} km/h"
            draw_label_with_bg(frame, trad_speed_text, (l, t-50), 
                              font_scale=0.5, bg_color=(100, 100, 200))
        
        # Draw motion trail (position history)
        if len(track_data['position_history']) > 1:
            for i in range(1, len(track_data['position_history'])):
                cv.line(frame, 
                        track_data['position_history'][i-1],
                        track_data['position_history'][i],
                        (0, 255, 255), 2)
    
    # Clean up old track data (remove tracks that haven't been seen for a while)
    current_track_ids = {t.track_id for t in tracks if t.is_confirmed()}
    old_track_ids = set(vehicle_data.keys()) - current_track_ids
    for old_id in old_track_ids:
        if len(vehicle_data[old_id]['timestamps']) > 0 and current_time - vehicle_data[old_id]['timestamps'][-1] > 2.0:
            vehicle_data.pop(old_id, None)
    
    # Display the frame
    cv.imshow("Vehicle Speed Tracking", cv.resize(frame, (800, 600)))
    
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()