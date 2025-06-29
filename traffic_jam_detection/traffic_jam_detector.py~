import cv2
import numpy as np
import matplotlib.pyplot as plt
from norfair.tracker import Detection, Tracker
from traffic_jam_detection.config import Config


class TrafficJamDetector:
    def __init__(self):
        self.tracker = Tracker(distance_function="euclidean", distance_threshold=30)

        self.selection_window_name = "Traffic Jam Detection - Select ROIs"

        # ROI selection variables
        self.polygons = []
        self.current_polygon = []
        self.polygon_np_arrays = []

        # Tracking variables - simplified like second code
        self.zone_stationary_counts = []
        self.zone_active_jams = []
        self.track_history = {}  # Stores last known 'cy' for each track_id

        # Evaluation variables
        self.jam_intervals = []
        self.jam_start_frames = []
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

    def _update_tracking_and_detect_jams(self, yolo_boxes, class_names, frame_idx):
        """
        Simplified version combining tracking and jam detection logic from second code.

        Args:
            yolo_boxes: YOLOv8 Result.boxes from Ultralytics model (result.boxes)
            class_names: Dict mapping class index to string label (e.g. {0: 'person', 1: 'car', ...})
            frame_idx: Current frame index

        Returns:
            List of updated tracked objects
        """
        # Process detections for Norfair tracker
        norfair_detections = []
        for box in yolo_boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = class_names.get(cls_id, None)

            if label in self.vehicle_labels:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                norfair_detections.append(
                    Detection(points=np.array([[cx, cy]]), scores=np.array([conf]))
                )

        # Update tracker
        tracked_objects = self.tracker.update(detections=norfair_detections)

        # Simplified jam detection logic (like second code)
        zone_stationary_this_frame = [0] * len(self.polygons)

        for obj in tracked_objects:
            cx, cy = map(int, obj.estimate[0])
            track_id = obj.id

            # Simple velocity calculation based on y-coordinate difference
            prev_y = self.track_history.get(track_id, cy)
            velocity = abs(cy - prev_y)
            self.track_history[track_id] = cy

            # Check which ROI the vehicle is in
            for idx, poly_np in enumerate(self.polygon_np_arrays):
                if cv2.pointPolygonTest(poly_np, (cx, cy), False) >= 0:
                    if velocity < Config.MIN_VELOCITY:
                        zone_stationary_this_frame[idx] += 1
                    break

        # Update jam detection for each zone (simplified logic from second code)
        for idx in range(len(self.polygons)):
            if zone_stationary_this_frame[idx] >= Config.JAM_THRESHOLD:
                self.zone_stationary_counts[idx] += 1
            else:
                # If jam was active and now not detected, record the interval
                if self.zone_active_jams[idx]:
                    end_frame = frame_idx - 1
                    start_frame = self.jam_start_frames[idx]
                    if start_frame != -1:
                        self.jam_intervals[idx].append((start_frame, end_frame))
                    self.jam_start_frames[idx] = -1
                self.zone_stationary_counts[idx] = 0

            # Check if jam should be predicted
            is_jam_predicted = self.zone_stationary_counts[idx] >= Config.JAM_FRAMES_DURATION

            # Track jam start
            if is_jam_predicted and not self.zone_active_jams[idx]:
                self.jam_start_frames[idx] = frame_idx

            self.zone_active_jams[idx] = is_jam_predicted

        return tracked_objects

    def _draw_visualization(self, frame, tracked_objects):
        """Draw visualization overlays - simplified version"""
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

        # Draw ROIs and jam status
        for idx, polygon in enumerate(self.polygons):
            is_jam_predicted = self.zone_active_jams[idx]
            color = (0, 0, 255) if is_jam_predicted else (0, 255, 0)

            cv2.polylines(frame, [self.polygon_np_arrays[idx]], isClosed=True, color=color, thickness=2)

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

    def finalize_jams(self, last_frame_idx):
        """Finalize any active jams at the end of processing"""
        for idx, active in enumerate(self.zone_active_jams):
            if active:
                start_frame = self.jam_start_frames[idx]
                if start_frame != -1:
                    self.jam_intervals[idx].append((start_frame, last_frame_idx))

    def save_jam_evaluation(self, fps, log_path="jam_evaluation.txt",
                            plot_path="jam_evaluation_plot.png"):
        """Saves jam detection results to a log file and generates a plot."""
        # Save to text file
        with open(log_path, "w") as f:
            for zone_idx, intervals in enumerate(self.jam_intervals):
                f.write(f"Zone {zone_idx + 1}:\n")
                if not intervals:
                    f.write("  No jams detected.\n")
                else:
                    for start_f, end_f in intervals:
                        start_s = round(start_f / fps, 2)
                        end_s = round(end_f / fps, 2)
                        f.write(f"  Jam from {start_s}s to {end_s}s\n")

        print(f"Jam evaluation saved to: {log_path}")

        # Generate plot
        plt.figure(figsize=(12, max(1, len(self.jam_intervals))))
        for i, intervals in enumerate(self.jam_intervals):
            for (start_f, end_f) in intervals:
                start_s = start_f / fps
                end_s = end_f / fps
                plt.barh(y=i, width=end_s - start_s, left=start_s, height=0.4, color="red")

        plt.yticks(range(len(self.jam_intervals)), [f"Zone {i + 1}" for i in range(len(self.jam_intervals))])
        plt.xlabel("Time (s)")
        plt.title("Traffic Jam Intervals per Zone (Model Prediction)")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Jam evaluation plot saved to: {plot_path}")