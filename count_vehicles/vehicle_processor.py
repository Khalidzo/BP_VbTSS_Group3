import cv2
import numpy as np
from typing import Any
from config import VEHICLE_CLASSES
from .config import TRACKER_CONFIG


class VehicleProcessor:
    def __init__(self):
        self.line_points = []
        self.tracker_config = TRACKER_CONFIG

        # Counters for all vehicle types
        self.car_out, self.car_in = 0, 0
        self.truck_out, self.truck_in = 0, 0
        self.motorcycle_out, self.motorcycle_in = 0, 0
        self.bus_out, self.bus_in = 0, 0

        self.already_counted = set()
        self.track_history = {}

        # Vehicle type mappings
        self.vehicle_types = VEHICLE_CLASSES

    def _draw_line_callback(self, event, x, y, flags, param):
        """Mouse callback for selecting counting line points"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.line_points) < 2:
            self.line_points.append((x, y))

    def _select_counting_line(self, frame):
        """Interactive selection of counting line"""
        temp_frame = frame.copy()
        window_name = "Select Line - Click 2 Points and Press ENTER"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._draw_line_callback)

        while True:
            display_frame = temp_frame.copy()

            # Draw selected points
            for pt in self.line_points:
                cv2.circle(display_frame, pt, 2, (0, 0, 255), -1)

            # Draw line if two points selected
            if len(self.line_points) == 2:
                cv2.line(
                    display_frame,
                    self.line_points[0],
                    self.line_points[1],
                    (0, 255, 0),
                    1,
                )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(self.line_points) == 2:  # ENTER
                print("Counting line confirmed.")
                break

        cv2.destroyWindow(window_name)

    def _process_detections(self, detections: Any, class_names: dict):
        """Process YOLO detections and update vehicle counts"""
        if detections.boxes is not None and detections.boxes.id is not None:
            for box, track_id in zip(detections.boxes, detections.boxes.id):
                cls_id = int(box.cls[0])
                label = class_names.get(cls_id, "unknown")

                if label in self.vehicle_types:
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_y = (y1 + y2) // 2
                    self._count_vehicles(track_id, label, current_y)

    def _count_vehicles(self, track_id: int, label: str, current_y: int):
        """Update vehicle counts based on crossing the counting line"""
        prev_y = self.track_history.get(track_id)

        if prev_y is not None and track_id not in self.already_counted:
            y1, y2 = self.line_points[0][1], self.line_points[1][1]
            line_y = (y1 + y2) // 2

            if prev_y < line_y <= current_y:
                # Vehicle moving downward (out)
                self._increment_counter(label, "out")
                self.already_counted.add(track_id)
            elif prev_y > line_y >= current_y:
                # Vehicle moving upward (in)
                self._increment_counter(label, "in")
                self.already_counted.add(track_id)

        self.track_history[track_id] = current_y

    def _increment_counter(self, label: str, direction: str):
        """Increment the appropriate counter based on vehicle type and direction"""
        counter_attr = f"{label}_{direction}"
        current_count = getattr(self, counter_attr, 0)
        setattr(self, counter_attr, current_count + 1)

    def _draw_annotations(self, frame: np.ndarray, detections: Any) -> np.ndarray:
        """Draw all visualizations on the frame"""
        # Draw counting line
        if len(self.line_points) == 2:
            cv2.line(frame, self.line_points[0], self.line_points[1], (0, 0, 255), 2)

        # Draw counters
        self._draw_counters(frame)

        return frame

    def _draw_counters(self, frame: np.ndarray):
        """Draw vehicle count information on frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Draw individual vehicle type counters
        counter_info = [
            (f"Cars Out: {self.car_out} | In: {self.car_in}", (255, 0, 0), 20),
            (f"Trucks Out: {self.truck_out} | In: {self.truck_in}", (0, 255, 0), 40),
            (
                f"Motorcycles Out: {self.motorcycle_out} | In: {self.motorcycle_in}",
                (0, 0, 255),
                60,
            ),
            (f"Buses Out: {self.bus_out} | In: {self.bus_in}", (255, 255, 0), 80),
        ]

        for text, color, y_pos in counter_info:
            cv2.putText(frame, text, (10, y_pos), font, font_scale, color, thickness)

        # Draw total count
        total_vehicles = self._calculate_total()
        cv2.putText(
            frame,
            f"Total: {total_vehicles}",
            (10, 125),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    def _calculate_total(self) -> int:
        """Calculate total number of counted vehicles"""
        return (
            self.car_out
            + self.car_in
            + self.truck_out
            + self.truck_in
            + self.motorcycle_out
            + self.motorcycle_in
            + self.bus_out
            + self.bus_in
        )

    def _save_log(self, log_path: str = "counter_log.txt"):
        """Save counting results to log file"""
        with open(log_path, "w") as f:
            f.write(f"Cars Out: {self.car_out}\nCars In: {self.car_in}\n")
            f.write(f"Trucks Out: {self.truck_out}\nTrucks In: {self.truck_in}\n")
            f.write(
                f"Motorcycles Out: {self.motorcycle_out}\nMotorcycles In: {self.motorcycle_in}\n"
            )
            f.write(f"Buses Out: {self.bus_out}\nBuses In: {self.bus_in}\n")
            f.write(f"Total: {self._calculate_total()}\n")

    def get_counts(self) -> dict:
        """Return dictionary with all vehicle counts"""
        return {
            "cars_out": self.car_out,
            "cars_in": self.car_in,
            "trucks_out": self.truck_out,
            "trucks_in": self.truck_in,
            "motorcycles_out": self.motorcycle_out,
            "motorcycles_in": self.motorcycle_in,
            "buses_out": self.bus_out,
            "buses_in": self.bus_in,
        }

    def print_summary(self):
        """Print detailed summary of counting results"""
        print("\n" + "=" * 50)
        print("VEHICLE COUNTING - SUMMARY")
        print("=" * 50)

        vehicle_summaries = [
            ("Cars", self.car_out, self.car_in),
            ("Trucks", self.truck_out, self.truck_in),
            ("Motorcycles", self.motorcycle_out, self.motorcycle_in),
            ("Buses", self.bus_out, self.bus_in),
        ]

        for vehicle_type, out_count, in_count in vehicle_summaries:
            total = out_count + in_count
            print(
                f"{vehicle_type:12s} Out: {out_count:3d} | In: {in_count:3d} | Total: {total:3d}"
            )

        print("-" * 50)

        total_out = self.car_out + self.truck_out + self.motorcycle_out + self.bus_out
        total_in = self.car_in + self.truck_in + self.motorcycle_in + self.bus_in
        total_all = self._calculate_total()

        print(
            f"TOTAL:       Out: {total_out:3d} | In: {total_in:3d} | Total: {total_all:3d}"
        )
        print("=" * 50)
