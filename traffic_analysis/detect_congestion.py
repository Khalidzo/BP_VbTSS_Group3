import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from collections import deque

# Import custom modules
from select_roi import ROISelector
from evaluate_performance import PerformanceEvaluator
from save_results import ResultsSaver


class TrafficJamDetector:
    def __init__(self, yolo_weights_path="Yolo-Weights/yolo12n.pt",
                 min_velocity=1.5,  # Balanced value
                 jam_threshold=3,  # Back to moderate value
                 jam_frames_duration=30,  # Faster response
                 target_width=640, target_height=360,
                 # Improved parameters for Ground Truth
                 ground_truth_jam_speed_threshold=1.8,  # Adjusted to min_velocity
                 ground_truth_jam_duration_frames=30,  # Consistent with model
                 # New parameters for improved detection
                 velocity_smoothing_frames=2,  # Less smoothing for faster response
                 min_vehicles_for_jam=3,  # Less strict
                 confidence_threshold=0.5,  # Less strict for more detections
                 # New adaptive parameters
                 stationary_ratio_threshold=0.5,  # 50% instead of 60%
                 velocity_variance_factor=0.7):  # More tolerance for velocity variance

        self.yolo_model = YOLO(yolo_weights_path)
        self.min_velocity = min_velocity
        self.jam_threshold = jam_threshold
        self.jam_frames_duration = jam_frames_duration
        self.target_width = target_width
        self.target_height = target_height

        self.tracker = Tracker(distance_function="euclidean", distance_threshold=30)

        # Initialize ROI selector
        self.roi_selector = ROISelector()
        self.polygons = []
        self.polygon_np_arrays = []

        self.zone_stationary_counts = []
        self.zone_active_jams = []
        self.track_history = {}  # Extended for velocity history

        # New attributes for improved detection
        self.velocity_smoothing_frames = velocity_smoothing_frames
        self.min_vehicles_for_jam = min_vehicles_for_jam
        self.confidence_threshold = confidence_threshold
        self.stationary_ratio_threshold = stationary_ratio_threshold
        self.velocity_variance_factor = velocity_variance_factor

        # Velocity history for smoothing
        self.track_velocity_history = {}  # track_id -> deque of velocities

        self.jam_intervals = []
        self.jam_start_frames = []

        self.ground_truth_jam_speed_threshold = ground_truth_jam_speed_threshold
        self.ground_truth_jam_duration_frames = ground_truth_jam_duration_frames
        self.evaluation_log = []

        self.vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}

        # Initialize performance evaluator
        self.performance_evaluator = PerformanceEvaluator(
            ground_truth_jam_speed_threshold,
            ground_truth_jam_duration_frames,
            min_vehicles_for_jam
        )

    def _calculate_smoothed_velocity(self, track_id, current_pos):
        """Calculates smoothed velocity over multiple frames."""
        if track_id not in self.track_history:
            self.track_history[track_id] = current_pos
            self.track_velocity_history[track_id] = deque(maxlen=self.velocity_smoothing_frames)
            return 0.0

        prev_pos = self.track_history[track_id]
        # Euclidean distance instead of just Y-coordinate
        velocity = np.sqrt((current_pos[0] - prev_pos[0]) ** 2 + (current_pos[1] - prev_pos[1]) ** 2)

        # Add velocity to history
        self.track_velocity_history[track_id].append(velocity)
        self.track_history[track_id] = current_pos

        # Calculate smoothed velocity
        if len(self.track_velocity_history[track_id]) > 0:
            return np.mean(list(self.track_velocity_history[track_id]))
        return velocity

    def _is_stationary_with_confidence(self, velocities_in_zone):
        """Extended stationary check with adaptive statistical measures."""
        if len(velocities_in_zone) == 0:
            return False

        mean_velocity = np.mean(velocities_in_zone)
        std_velocity = np.std(velocities_in_zone)

        # Adaptive thresholds based on vehicle count
        vehicle_count = len(velocities_in_zone)

        # Basic condition: average velocity below threshold
        is_low_speed = mean_velocity < self.min_velocity

        # Adaptive variance tolerance (more tolerance for fewer vehicles)
        variance_threshold = self.min_velocity * self.velocity_variance_factor
        if vehicle_count <= 2:
            variance_threshold *= 1.5  # More tolerance for few vehicles

        is_consistent = std_velocity < variance_threshold

        # Additional condition: proportion of very slow vehicles
        very_slow_vehicles = sum(1 for v in velocities_in_zone if v < self.min_velocity * 0.8)
        slow_ratio = very_slow_vehicles / len(velocities_in_zone)

        # At least 40% of vehicles must be very slow
        has_sufficient_slow_vehicles = slow_ratio >= 0.4

        return is_low_speed and (is_consistent or has_sufficient_slow_vehicles)

    def run(self, video_path):
        """Runs the main traffic jam detection process on a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file '{video_path}'.")
            return

        success, frame = cap.read()
        if not success:
            print("❌ Error: Could not read the first frame from the video.")
            cap.release()
            return

        frame = cv2.resize(frame, (self.target_width, self.target_height))

        # Use ROI selector to get regions of interest
        self.polygons, self.polygon_np_arrays = self.roi_selector.select_rois(frame)

        # Initialize zone tracking lists
        self.zone_stationary_counts = [0] * len(self.polygons)
        self.zone_active_jams = [False] * len(self.polygons)
        self.jam_intervals = [[] for _ in range(len(self.polygons))]
        self.jam_start_frames = [-1] * len(self.polygons)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("\n--- Starting Traffic Jam Detection ---")
        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            current_frame_display = cv2.resize(frame, (self.target_width, self.target_height))

            # YOLO prediction with confidence threshold
            results_boxes = self.yolo_model.predict(current_frame_display, verbose=False)[0].boxes

            norfair_detections = []
            for box in results_boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.yolo_model.names[cls_id]

                # Only use detections with sufficient confidence
                if label in self.vehicle_labels and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    norfair_detections.append(Detection(points=np.array([[cx, cy]]), scores=np.array([conf])))

            tracked_objects = self.tracker.update(detections=norfair_detections)

            # Initialization for current frame
            zone_vehicle_counts = [0] * len(self.polygons)
            zone_velocities_this_frame = [[] for _ in self.polygons]
            zone_stationary_vehicles = [0] * len(self.polygons)

            for obj in tracked_objects:
                cx, cy = map(int, obj.estimate[0])
                track_id = obj.id

                # Improved velocity calculation
                smoothed_velocity = self._calculate_smoothed_velocity(track_id, (cx, cy))

                # Check for each ROI
                is_inside_any_roi = False
                for idx, poly_np in enumerate(self.polygon_np_arrays):
                    if cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0:
                        is_inside_any_roi = True
                        zone_vehicle_counts[idx] += 1
                        zone_velocities_this_frame[idx].append(smoothed_velocity)

                        if smoothed_velocity < self.min_velocity:
                            zone_stationary_vehicles[idx] += 1
                        break

                # Improved visualization
                track_color = (255, 0, 0) if is_inside_any_roi else (0, 255, 255)
                cv2.circle(current_frame_display, (cx, cy), 4, track_color, -1)
                # Display velocity
                cv2.putText(current_frame_display, "",
                            (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, track_color, 1)

            # Improved jam detection for each zone
            frame_roi_data_for_log = []
            for idx, polygon in enumerate(self.polygons):
                # More flexible logic: minimum number of vehicles + adaptive evaluation
                has_min_vehicles = zone_vehicle_counts[idx] >= self.min_vehicles_for_jam

                jam_detected_this_frame = False

                if zone_velocities_this_frame[idx]:  # Vehicles present in zone
                    is_zone_congested = self._is_stationary_with_confidence(zone_velocities_this_frame[idx])

                    if has_min_vehicles:
                        # Standard logic: enough vehicles + velocity analysis
                        stationary_ratio = zone_stationary_vehicles[idx] / zone_vehicle_counts[idx]
                        if is_zone_congested and stationary_ratio >= self.stationary_ratio_threshold:
                            jam_detected_this_frame = True
                    else:
                        # Few vehicles: stricter criteria
                        if (is_zone_congested and
                                zone_stationary_vehicles[idx] >= max(1, zone_vehicle_counts[idx] - 1)):
                            jam_detected_this_frame = True

                # Jam counter update
                if jam_detected_this_frame:
                    self.zone_stationary_counts[idx] += 1
                else:
                    # End jam with hysteresis (smooth transition)
                    if self.zone_stationary_counts[idx] > 0:
                        self.zone_stationary_counts[idx] = max(0, self.zone_stationary_counts[idx] - 2)

                    if self.zone_active_jams[idx] and self.zone_stationary_counts[idx] == 0:
                        end_frame = frame_idx - 1
                        start_frame = self.jam_start_frames[idx]
                        if start_frame != -1:
                            self.jam_intervals[idx].append((start_frame, end_frame))
                        self.jam_start_frames[idx] = -1

                is_jam_predicted = self.zone_stationary_counts[idx] >= self.jam_frames_duration

                # Track jam start
                if is_jam_predicted and not self.zone_active_jams[idx]:
                    self.jam_start_frames[idx] = frame_idx
                self.zone_active_jams[idx] = is_jam_predicted

                # Visualization with more information
                color = (0, 0, 255) if is_jam_predicted else (0, 255, 0)
                cv2.polylines(current_frame_display, [self.polygon_np_arrays[idx]], isClosed=True, color=color,
                              thickness=2)

                status_text = "Traffic Jam" if is_jam_predicted else "Traffic Flowing"
                info_text = f"Zone {idx + 1}: {status_text}"

                cv2.putText(current_frame_display, info_text,
                            (polygon[0][0], polygon[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                # Collect data for evaluation
                avg_velocity_in_zone = None
                if zone_velocities_this_frame[idx]:
                    avg_velocity_in_zone = np.mean(zone_velocities_this_frame[idx])

                frame_roi_data_for_log.append({
                    "zone_idx": idx,
                    "predicted_jam": is_jam_predicted,
                    "average_velocity": avg_velocity_in_zone,
                    "vehicle_count": zone_vehicle_counts[idx],
                    "stationary_count": zone_stationary_vehicles[idx]
                })

            self.evaluation_log.append({
                "frame_idx": frame_idx,
                "roi_data": frame_roi_data_for_log
            })

            cv2.imshow("Traffic Jam Detection", current_frame_display)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        # Close open jams at the end
        last_frame_idx = frame_idx
        for idx, active in enumerate(self.zone_active_jams):
            if active:
                start_frame = self.jam_start_frames[idx]
                if start_frame != -1:
                    self.jam_intervals[idx].append((start_frame, last_frame_idx))

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        cv2.destroyAllWindows()

        # Save results using ResultsSaver
        ResultsSaver.save_jam_evaluation(self.jam_intervals, fps)

        # Evaluate performance using PerformanceEvaluator
        self.performance_evaluator.calculate_ground_truth_and_evaluate(
            self.evaluation_log, len(self.polygons), fps
        )

        print("\n--- Traffic Jam Detection Finished ---")


if __name__ == "__main__":
    # Balanced configuration for better balance between precision and recall
    detector = TrafficJamDetector(
        min_velocity=1.5,  # Balanced between too strict and too permissive
        jam_threshold=3,  # Moderate number of stationary vehicles
        jam_frames_duration=30,  # Faster response to traffic jams
        ground_truth_jam_speed_threshold=1.8,  # Consistent with min_velocity
        ground_truth_jam_duration_frames=30,  # Consistent with model
        velocity_smoothing_frames=2,  # Less smoothing for faster response
        min_vehicles_for_jam=3,  # Less strict
        confidence_threshold=0.5,  # Allow more detections
        stationary_ratio_threshold=0.5,  # 50% instead of 60%
        velocity_variance_factor=0.7  # More tolerance for velocity variance
    )

    detector.run(video_path="Videos/eva7.mp4")