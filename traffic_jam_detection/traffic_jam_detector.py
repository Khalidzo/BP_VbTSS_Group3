import cv2
import numpy as np
from norfair.tracker import Detection, Tracker
from collections import deque
from traffic_jam_detection.config import Config



class TrafficJamDetector:
    def __init__(self):
        self.tracker = Tracker(distance_function="euclidean", distance_threshold=30)

        self.selection_window_name = "Traffic Jam Detection - Select ROIs"

        # ROI selection variables
        self.current_polygon = []
        self.polygons = []
        self.polygon_np_arrays = []

        # Tracking variables
        self.zone_stationary_counts = []
        self.zone_active_jams = []
        self.track_history = {}
        self.track_velocity_history = {}

        # Evaluation variables
        self.jam_intervals = []
        self.jam_start_frames = []
        self.evaluation_log = []
        self.vehicle_labels = {"car", "truck", "bus", "motorbike", "bicycle"}

    def _draw_polygon_callback(self, event, x, y, flags, param):
        """Callback function for mouse events during ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))
            print(f"Added point: ({x}, {y})")

    def _select_rois(self, frame):
        """ROI selection similar to code 1's roi_selector"""
        display_frame = frame.copy()
        cv2.namedWindow(self.selection_window_name)
        cv2.setMouseCallback(self.selection_window_name, self._draw_polygon_callback)

        print("\n--- 🚦Traffic Jam Detection - ROI Selection ---")
        print("🖱️  Click to define polygon points.")
        print("↩️  Press ENTER to confirm the current polygon.")
        print("❌ Press ESC to finish ROI selection.\n")

        while True:
            temp_display_frame = display_frame.copy()

            # Draw existing polygons
            for poly_np in self.polygon_np_arrays:
                cv2.polylines(
                    temp_display_frame,
                    [poly_np],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )

            # Draw current polygon being created
            if len(self.current_polygon) >= 2:
                cv2.polylines(
                    temp_display_frame,
                    [np.array(self.current_polygon, np.int32)],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=1,
                )

            if len(self.current_polygon) >= 3:
                cv2.polylines(
                    temp_display_frame,
                    [np.array(self.current_polygon, np.int32)],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=1,
                )

            for pt in self.current_polygon:
                cv2.circle(temp_display_frame, pt, 3, (0, 0, 255), -1)

            cv2.imshow("Traffic Jam Detection - Select ROIs", temp_display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER key
                if len(self.current_polygon) >= 3:
                    self.polygons.append(self.current_polygon.copy())
                    self.polygon_np_arrays.append(
                        np.array(self.current_polygon, np.int32)
                    )
                    self.zone_stationary_counts.append(0)
                    self.zone_active_jams.append(False)
                    self.jam_intervals.append([])
                    self.jam_start_frames.append(-1)
                    print(f"Polygon {len(self.polygons)} saved.")
                    self.current_polygon.clear()
                else:
                    print("Minimum 3 points required to save polygon.")
            elif key == 27:  # ESC key
                break
        
        print(f"\n✅ ROI Selection complete. Total zones defined: {len(self.polygons)}")
        cv2.destroyWindow(self.selection_window_name)

    def _calculate_smoothed_velocity(self, track_id, current_pos):
        """Calculate smoothed velocity over multiple frames"""
        if track_id not in self.track_history:
            self.track_history[track_id] = current_pos
            self.track_velocity_history[track_id] = deque(
                maxlen=Config.VELOCITY_SMOOTHING_FRAMES
            )
            return 0.0

        prev_pos = self.track_history[track_id]
        velocity = np.sqrt(
            (current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2
        )

        self.track_velocity_history[track_id].append(velocity)
        self.track_history[track_id] = current_pos

        if len(self.track_velocity_history[track_id]) > 0:
            return np.mean(list(self.track_velocity_history[track_id]))
        return velocity

    def _is_stationary_with_confidence(self, velocities_in_zone):
        """Check if vehicles in zone are stationary with confidence measures"""
        if len(velocities_in_zone) == 0:
            return False

        mean_velocity = np.mean(velocities_in_zone)
        std_velocity = np.std(velocities_in_zone)
        vehicle_count = len(velocities_in_zone)

        # Basic condition: average velocity below threshold
        is_low_speed = mean_velocity < Config.MIN_VELOCITY

        # Adaptive variance tolerance
        variance_threshold = Config.MIN_VELOCITY * Config.VELOCITY_VARIANCE_FACTOR
        if vehicle_count <= 2:
            variance_threshold *= 1.5
        is_consistent = std_velocity < variance_threshold

        # Proportion of very slow vehicles
        very_slow_vehicles = sum(
            1 for v in velocities_in_zone if v < Config.MIN_VELOCITY * 0.8
        )
        slow_ratio = very_slow_vehicles / len(velocities_in_zone)
        has_sufficient_slow_vehicles = slow_ratio >= 0.4

        return is_low_speed and (is_consistent or has_sufficient_slow_vehicles)

    def _process_detections_and_tracking(self, yolo_boxes, class_names):
        """
        Process external YOLO detection boxes and update Norfair tracker.

        Args:
            yolo_boxes: YOLOv8 Result.boxes from Ultralytics model (result.boxes)
            class_names: Dict mapping class index to string label (e.g. {0: 'person', 1: 'car', ...})

        Returns:
            List of updated tracked objects
        """
        norfair_detections = []

        for box in yolo_boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = class_names.get(cls_id, None)

            if label in self.vehicle_labels and conf >= Config.CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                norfair_detections.append(
                    Detection(points=np.array([[cx, cy]]), scores=np.array([conf]))
                )

        return self.tracker.update(detections=norfair_detections)

    def _process_roi_analysis(self, tracked_objects, frame_idx):
        """Analyze ROIs for jam detection - similar to code 1's roi processing"""
        zone_vehicle_counts = [0] * len(self.polygons)
        zone_velocities_this_frame = [[] for _ in self.polygons]
        zone_stationary_vehicles = [0] * len(self.polygons)

        for obj in tracked_objects:
            cx, cy = map(int, obj.estimate[0])
            track_id = obj.id

            smoothed_velocity = self._calculate_smoothed_velocity(track_id, (cx, cy))

            # Check which ROI the vehicle is in
            for idx, poly_np in enumerate(self.polygon_np_arrays):
                if cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0:
                    zone_vehicle_counts[idx] += 1
                    zone_velocities_this_frame[idx].append(smoothed_velocity)
                    if smoothed_velocity < Config.MIN_VELOCITY:
                        zone_stationary_vehicles[idx] += 1
                    break

        return zone_vehicle_counts, zone_velocities_this_frame, zone_stationary_vehicles

    def _update_jam_detection(
        self,
        zone_vehicle_counts,
        zone_velocities_this_frame,
        zone_stationary_vehicles,
        frame_idx,
    ):
        """Update jam detection status for each zone"""
        frame_roi_data_for_log = []

        for idx in range(len(self.polygons)):
            has_min_vehicles = zone_vehicle_counts[idx] >= Config.MIN_VEHICLES_FOR_JAM
            jam_detected_this_frame = False

            if zone_velocities_this_frame[idx]:  # Vehicles present in zone
                is_zone_congested = self._is_stationary_with_confidence(
                    zone_velocities_this_frame[idx]
                )

                if has_min_vehicles:
                    stationary_ratio = (
                        zone_stationary_vehicles[idx] / zone_vehicle_counts[idx]
                    )
                    if (
                        is_zone_congested
                        and stationary_ratio >= Config.STATIONARY_RATIO_THRESHOLD
                    ):
                        jam_detected_this_frame = True
                else:
                    # Few vehicles: stricter criteria
                    if is_zone_congested and zone_stationary_vehicles[idx] >= max(
                        1, zone_vehicle_counts[idx] - 1
                    ):
                        jam_detected_this_frame = True

            # Update jam counter
            if jam_detected_this_frame:
                self.zone_stationary_counts[idx] += 1
            else:
                if self.zone_stationary_counts[idx] > 0:
                    self.zone_stationary_counts[idx] = max(
                        0, self.zone_stationary_counts[idx] - 2
                    )

                if self.zone_active_jams[idx] and self.zone_stationary_counts[idx] == 0:
                    end_frame = frame_idx - 1
                    start_frame = self.jam_start_frames[idx]
                    if start_frame != -1:
                        self.jam_intervals[idx].append((start_frame, end_frame))
                    self.jam_start_frames[idx] = -1

            is_jam_predicted = (
                self.zone_stationary_counts[idx] >= Config.JAM_FRAMES_DURATION
            )

            # Track jam start
            if is_jam_predicted and not self.zone_active_jams[idx]:
                self.jam_start_frames[idx] = frame_idx

            self.zone_active_jams[idx] = is_jam_predicted

            # Collect data for evaluation
            avg_velocity_in_zone = None
            if zone_velocities_this_frame[idx]:
                avg_velocity_in_zone = np.mean(zone_velocities_this_frame[idx])

            frame_roi_data_for_log.append(
                {
                    "zone_idx": idx,
                    "predicted_jam": is_jam_predicted,
                    "average_velocity": avg_velocity_in_zone,
                    "vehicle_count": zone_vehicle_counts[idx],
                    "stationary_count": zone_stationary_vehicles[idx],
                }
            )

        self.evaluation_log.append(
            {"frame_idx": frame_idx, "roi_data": frame_roi_data_for_log}
        )

    def _draw_visualization(self, frame, tracked_objects):
        """Draw visualization overlays - similar to code 1's draw functions"""
        # Draw tracked objects
        for obj in tracked_objects:
            cx, cy = map(int, obj.estimate[0])

            # Check if inside any ROI
            is_inside_any_roi = False
            for poly_np in self.polygon_np_arrays:
                if cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0:
                    is_inside_any_roi = True
                    break

            track_color = (255, 0, 0) if is_inside_any_roi else (0, 255, 255)
            cv2.circle(frame, (cx, cy), 5, track_color, -1)

        # Draw ROIs and jam status
        for idx, (polygon, poly_np) in enumerate(
            zip(self.polygons, self.polygon_np_arrays)
        ):
            is_jam_predicted = self.zone_active_jams[idx]
            color = (0, 0, 255) if is_jam_predicted else (0, 255, 0)

            cv2.polylines(frame, [poly_np], isClosed=True, color=color, thickness=2)

            status_text = "Traffic Jam" if is_jam_predicted else "Traffic Flowing"
            info_text = f"Zone {idx + 1}: {status_text}"
            cv2.putText(
                frame,
                info_text,
                (polygon[0][0], polygon[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
