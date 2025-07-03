import cv2
from ultralytics import YOLO


class VehicleCounter:
    def __init__(self, yolo_weights, tracker_config=None):
        self.yolo_model = YOLO(yolo_weights)
        self.tracker_config = tracker_config or {
            "conf": 0.5,
            "iou": 0.5,
            "tracker": "bytetrack.yaml",
        }
        self.line_points = []
        self.line_position = None

        # Counters for all vehicle types
        self.car_out, self.car_in = 0, 0
        self.truck_out, self.truck_in = 0, 0
        self.motorcycle_out, self.motorcycle_in = 0, 0
        self.bus_out, self.bus_in = 0, 0

        self.already_counted = set()
        self.track_history = {}
        self.target_width = 640
        self.target_height = 360

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        return cap

    def _resize_frame(self, frame):
        return cv2.resize(frame, (self.target_width, self.target_height))

    def _track_vehicles(self, frame):
        results = self.yolo_model.track(frame, persist=True, **self.tracker_config)[0]
        return results

    def _draw_line_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.line_points) < 2:
            self.line_points.append((x, y))

    def _select_counting_line(self, frame):
        temp_frame = frame.copy()
        cv2.namedWindow("Select Line - Click 2 Points and Press ENTER")
        cv2.setMouseCallback(
            "Select Line - Click 2 Points and Press ENTER", self._draw_line_callback
        )

        while True:
            display_frame = temp_frame.copy()
            for pt in self.line_points:
                cv2.circle(display_frame, pt, 2, (0, 0, 255), -1)
            if len(self.line_points) == 2:
                cv2.line(
                    display_frame,
                    self.line_points[0],
                    self.line_points[1],
                    (0, 255, 0),
                    1,
                )

            cv2.imshow("Select Line - Click 2 Points and Press ENTER", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13 and len(self.line_points) == 2:  # ENTER
                print("Line confirmed.")
                break

        cv2.destroyWindow("Select Line - Click 2 Points and Press ENTER")

    def _count_vehicles(self, track_id, label, current_x, current_y):
        prev_pos = self.track_history.get(track_id)

        if prev_pos is not None and track_id not in self.already_counted:
            prev_x, prev_y = prev_pos

            # Calculate the counting line
            x1, y1 = self.line_points[0]
            x2, y2 = self.line_points[1]

            # Check if the vehicle trajectory crosses the line
            if self._line_intersection(prev_x, prev_y, current_x, current_y, x1, y1, x2, y2):
                # Determine the direction based on the line
                if self._is_crossing_out(prev_x, prev_y, current_x, current_y, x1, y1, x2, y2):
                    # Vehicle moving out
                    if label == "car":
                        self.car_out += 1
                    elif label == "truck":
                        self.truck_out += 1
                    elif label == "motorcycle":
                        self.motorcycle_out += 1
                    elif label == "bus":
                        self.bus_out += 1
                    self.already_counted.add(track_id)
                else:
                    # Vehicle moving in
                    if label == "car":
                        self.car_in += 1
                    elif label == "truck":
                        self.truck_in += 1
                    elif label == "motorcycle":
                        self.motorcycle_in += 1
                    elif label == "bus":
                        self.bus_in += 1
                    self.already_counted.add(track_id)

        self.track_history[track_id] = (current_x, current_y)

    def _line_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Checks if two lines intersect"""
        # Line 1: (x1,y1) to (x2,y2) - Vehicle trajectory
        # Line 2: (x3,y3) to (x4,y4) - Counting line

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False  # Lines are parallel

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        return 0 <= t <= 1 and 0 <= u <= 1

    def _is_crossing_out(self, prev_x, prev_y, curr_x, curr_y, line_x1, line_y1, line_x2, line_y2):
        """Determines the crossing direction based on the line"""
        # Calculate the normal vector of the line
        dx = line_x2 - line_x1
        dy = line_y2 - line_y1

        # Normal vector (pointing right)
        nx = -dy
        ny = dx

        # Movement vector of the vehicle
        move_x = curr_x - prev_x
        move_y = curr_y - prev_y

        # Dot product to determine direction
        dot_product = move_x * nx + move_y * ny

        return dot_product > 0  # True = "out", False = "in"

    def _draw_annotations(self, frame, results):
        if len(self.line_points) == 2:
            cv2.line(frame, self.line_points[0], self.line_points[1], (0, 0, 255), 2)

        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                cls_id = int(box.cls[0])
                label = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                if label in ["car", "truck", "motorcycle", "bus"]:
                    # Different colors for different vehicle types
                    color_map = {
                        "car": (255, 0, 0),  # Blue
                        "truck": (0, 255, 0),  # Green
                        "motorcycle": (0, 0, 255),  # Red
                        "bus": (255, 255, 0),  # Cyan
                    }
                    color = color_map.get(label, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(
                        frame,
                        f"{label}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

        # Extended display of counter states
        cv2.putText(
            frame,
            f"Cars Out: {self.car_out} | In: {self.car_in}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            1,
        )
        cv2.putText(
            frame,
            f"Trucks Out: {self.truck_out} | In: {self.truck_in}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            f"Motorcycles Out: {self.motorcycle_out} | In: {self.motorcycle_in}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Buses Out: {self.bus_out} | In: {self.bus_in}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
        )

        # Display total count
        total_vehicles = (
            self.car_out
            + self.car_in
            + self.truck_out
            + self.truck_in
            + self.motorcycle_out
            + self.motorcycle_in
            + self.bus_out
            + self.bus_in
        )
        cv2.putText(
            frame,
            f"Total: {total_vehicles}",
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        return frame

    def _save_log(self, log_path="counter_log.txt"):
        with open(log_path, "w") as f:
            f.write(f"Cars Out: {self.car_out}\nCars In: {self.car_in}\n")
            f.write(f"Trucks Out: {self.truck_out}\nTrucks In: {self.truck_in}\n")
            f.write(
                f"Motorcycles Out: {self.motorcycle_out}\nMotorcycles In: {self.motorcycle_in}\n"
            )
            f.write(f"Buses Out: {self.bus_out}\nBuses In: {self.bus_in}\n")
            f.write(
                f"Total: {self.car_out + self.car_in + self.truck_out + self.truck_in + self.motorcycle_out + self.motorcycle_in + self.bus_out + self.bus_in}\n"
            )

    def count_vehicles(self, video_path):
        cap = self._load_video(video_path)

        success, first_frame = cap.read()
        if not success:
            print("Error reading the first frame to initialize the counting line.")
            cap.release()
            cv2.destroyAllWindows()
            return

        first_frame_resized = self._resize_frame(first_frame)
        self._select_counting_line(first_frame_resized)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = self._resize_frame(frame)
            results = self._track_vehicles(frame)

            if results.boxes is not None and results.boxes.id is not None:
                for box, track_id in zip(results.boxes, results.boxes.id):
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names[cls_id]
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    if label in ["car", "truck", "motorcycle", "bus"]:
                        track_id = int(track_id)
                        current_x = (x1 + x2) // 2
                        current_y = (y1 + y2) // 2
                        self._count_vehicles(track_id, label, current_x, current_y)

            annotated_frame = self._draw_annotations(frame, results)

            cv2.imshow("Vehicle Counting", annotated_frame)
            if cv2.waitKey(1) == 27:  # ESC to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        self._save_log()

    def get_counts(self):
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
        """Prints a summary statistics"""
        print("\n" + "=" * 50)
        print("VEHICLE COUNTING - SUMMARY")
        print("=" * 50)
        print(
            f"Cars:        Out: {self.car_out:3d} | In: {self.car_in:3d} | Total: {self.car_out + self.car_in:3d}"
        )
        print(
            f"Trucks:      Out: {self.truck_out:3d} | In: {self.truck_in:3d} | Total: {self.truck_out + self.truck_in:3d}"
        )
        print(
            f"Motorcycles: Out: {self.motorcycle_out:3d} | In: {self.motorcycle_in:3d} | Total: {self.motorcycle_out + self.motorcycle_in:3d}"
        )
        print(
            f"Buses:       Out: {self.bus_out:3d} | In: {self.bus_in:3d} | Total: {self.bus_out + self.bus_in:3d}"
        )
        print("-" * 50)
        total = (
            self.car_out
            + self.car_in
            + self.truck_out
            + self.truck_in
            + self.motorcycle_out
            + self.motorcycle_in
            + self.bus_out
            + self.bus_in
        )
        print(
            f"TOTAL:       Out: {self.car_out + self.truck_out + self.motorcycle_out + self.bus_out:3d} | In: {self.car_in + self.truck_in + self.motorcycle_in + self.bus_in:3d} | Total: {total:3d}"
        )
        print("=" * 50)


if __name__ == "__main__":
    tracker_config = {"conf": 0.5, "iou": 0.5, "tracker": "bytetrack.yaml"}

    counter = VehicleCounter(
        yolo_weights="Yolo-Weights/yolo12n.pt", tracker_config=tracker_config
    )
    counter.count_vehicles(video_path="Videos/highway.mp4")

    # Detailed output of results
    counter.print_summary()

    counts = counter.get_counts()
    print(f"\nReturn Dictionary: {counts}")
