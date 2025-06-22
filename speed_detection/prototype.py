import cv2 as cv
import numpy as np
from ultralytics import YOLO
import math

"""
Vehicle distance + speed estimator using a per-track constant-velocity Kalman filter.
The filter runs in the horizontal ground plane (X-Z) and outputs a far less jittery
speed (km/h) than simple frame-to-frame differencing.
"""

# ───────── Camera parameters ─────────
focal_length = 862.06  # pixels
roll = -0.48  # degrees
pitch = -19.61  # degrees
camera_height = 6.0  # metres

# ───────── Tracker / KF hyper-parameters ─────────
MATCH_THRESHOLD_METERS = 3.0  # max 3-D distance to associate detection → track
MAX_LOST_FRAMES = 20  # frames before a missing track is dropped
PROCESS_VAR = 1.0  # Q (m² / s⁴)
MEASUREMENT_VAR = 1.0  # R (m²)
INITIAL_POS_VAR = 25.0  # P₀ position var (m²)
INITIAL_VEL_VAR = 25.0  # P₀ velocity var ((m/s)²)
CONF_THRESH = 0.1  # YOLO confidence threshold

# ───────── Load video & YOLO ─────────
video_path = "D:\\2016-ITS-BrnoCompSpeed\\dataset\\session6_center\\video.avi"
model = YOLO("yolo11n.pt")
cap = cv.VideoCapture(video_path)

fps = cap.get(cv.CAP_PROP_FPS) or 30
frame_interval = 1.0 / fps

cv.namedWindow("frame", cv.WINDOW_NORMAL)

# ───────── COCO vehicle classes ─────────
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ───────── First frame – derive principal point (centred) ─────────
ok, frame = cap.read()
if not ok:
    raise RuntimeError("Could not read video")

h0, w0 = frame.shape[:2]
cx, cy = w0 / 2, h0 / 2

# For display scaling only
scaled_w, scaled_h = 800, 600

# ───────── Camera intrinsics & extrinsics ─────────
K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=float)

roll_rad, pitch_rad = map(math.radians, (roll, pitch))
R_roll = np.array(
    [
        [math.cos(roll_rad), -math.sin(roll_rad), 0],
        [math.sin(roll_rad), math.cos(roll_rad), 0],
        [0, 0, 1],
    ],
    float,
)
R_pitch = np.array(
    [
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad), math.cos(pitch_rad)],
    ],
    float,
)
R = R_roll @ R_pitch  # first pitch, then roll


def project_to_ground(img_pt):
    """Back-project 2-D image point to ground plane (Y = 0)."""
    u, v = img_pt
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_cam[1] *= -1  # y-down → y-up
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = R.T @ ray_cam

    if abs(ray_world[1]) < 1e-5:
        return None  # Avoid division by zero or near-flat rays

    t = -camera_height / ray_world[1]
    if t < 0:
        return None  # Point would be behind the camera

    pos3d = np.array([0, camera_height, 0]) + t * ray_world  # [X, Y=0, Z]

    # ✅ Print the 3D distance from camera to the projected ground point
    dist = np.linalg.norm(pos3d)
    print(f"Projected point at {pos3d}, distance to camera = {dist:.2f} meters")

    return pos3d


def horizontal_distance(p1, p2):
    return np.linalg.norm(p1[[0, 2]] - p2[[0, 2]])


# ───────── Kalman-filter track class ─────────
class Track:
    _F_base = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    _H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    _R = np.eye(2) * MEASUREMENT_VAR

    def __init__(self, track_id, pos3d, bbox, timestamp):
        x, z = pos3d[[0, 2]]
        self.id = track_id
        self.state = np.array([x, z, 0, 0], float)  # [x, z, vx, vz]
        self.P = np.diag(
            [INITIAL_POS_VAR, INITIAL_POS_VAR, INITIAL_VEL_VAR, INITIAL_VEL_VAR]
        )
        self.bbox = bbox
        self.prev_t = timestamp
        self.lost = 0

    # KF helpers
    @staticmethod
    def _Q(dt):
        q = PROCESS_VAR
        dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
        return (
            np.array(
                [
                    [dt4 / 4, 0, dt3 / 2, 0],
                    [0, dt4 / 4, 0, dt3 / 2],
                    [dt3 / 2, 0, dt2, 0],
                    [0, dt3 / 2, 0, dt2],
                ],
                float,
            )
            * q
        )

    def _predict(self, dt):
        F = self._F_base.copy()
        F[0, 2] = F[1, 3] = dt
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self._Q(dt)

    def _update(self, z):
        y = z - self._H @ self.state
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(4) - K @ self._H) @ self.P

    # public
    def step(self, pos3d, bbox, timestamp):
        dt = timestamp - self.prev_t
        if dt > 0:
            self._predict(dt)
            self._update(pos3d[[0, 2]])
        self.prev_t = timestamp
        self.bbox = bbox
        self.lost = 0

    @property
    def speed_kmh(self):
        vx, vz = self.state[2], self.state[3]
        return math.hypot(vx, vz) * 3.6

    @property
    def pos3d(self):
        return np.array([self.state[0], 0, self.state[1]])


# ───────── Tracking loop ─────────
next_id = 0
tracks = {}
frame_idx = 0
cap.set(cv.CAP_PROP_POS_FRAMES, 0)


def bbox_outside(b, w, h):
    x1, y1, x2, y2 = b
    return x2 < 0 or x1 > w or y2 < 0 or y1 > h


while True:
    ok, frame = cap.read()
    if not ok:
        break
    t_now = frame_idx * frame_interval
    frame_idx += 1

    # ── DETECTION ──
    dets = []
    for box in model(frame, verbose=False)[0].boxes:
        cls, conf = int(box.cls[0]), float(box.conf[0])
        if cls in VEHICLE_CLASSES and conf >= CONF_THRESH:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bc = (int((x1 + x2) / 2), y2)  # bottom-centre pixel
            pos3d = project_to_ground(bc)
            if pos3d is not None:
                dets.append({"bbox": (x1, y1, x2, y2), "pos3d": pos3d})

    # ── GREEDY GROUND-PLANE ASSOCIATION ──
    un_dets = list(range(len(dets)))
    un_trks = list(tracks.keys())
    dists = np.full((len(un_dets), len(un_trks)), np.inf)
    for i, di in enumerate(un_dets):
        for j, tj in enumerate(un_trks):
            dists[i, j] = horizontal_distance(dets[di]["pos3d"], tracks[tj].pos3d)

    matches = []
    while dists.size:
        i, j = divmod(dists.argmin(), dists.shape[1])
        i, j = int(i), int(j)
        if dists[i, j] > MATCH_THRESHOLD_METERS:  # type: ignore
            break
        di, tj = un_dets.pop(i), un_trks.pop(j)
        matches.append((di, tj))
        dists = np.delete(np.delete(dists, i, 0), j, 1)  # type: ignore

    # ── UPDATE / CREATE ──
    for di, tj in matches:
        d = dets[di]
        tracks[tj].step(d["pos3d"], d["bbox"], t_now)

    for di in un_dets:
        d = dets[di]
        tracks[next_id] = Track(next_id, d["pos3d"], d["bbox"], t_now)
        next_id += 1

    # ── MANAGE LOST ──
    h, w = frame.shape[:2]
    for tid in list(tracks.keys()):
        if tid not in [tj for _, tj in matches]:
            trk = tracks[tid]
            trk.lost += 1
            # Optional: drop immediately if last bbox is out of view
            if trk.lost > 0 and bbox_outside(trk.bbox, w, h):
                del tracks[tid]
                continue
            if trk.lost > MAX_LOST_FRAMES:
                del tracks[tid]

    # ── VISUALISATION ──
    for trk in tracks.values():
        if trk.lost:  # skip drawing unmatched tracks
            continue
        x1, y1, x2, y2 = trk.bbox
        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.putText(
            frame,
            f"ID {trk.id}  {trk.speed_kmh:.1f} km/h",
            (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    # Draw principal point
    cv.circle(frame, (int(cx), int(cy)), 6, (0, 255, 255), -1)
    cv.putText(
        frame,
        "cx,cy",
        (int(cx) + 10, int(cy) - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    scaled = cv.resize(frame, (scaled_w, scaled_h))
    cv.imshow("frame", scaled)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
