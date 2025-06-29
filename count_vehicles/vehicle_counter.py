import numpy as np
import cv2
from base_feature import VideoFeatureProcessor
from config import TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT
from count_vehicles.vehicle_processor import VehicleProcessor
from typing import Any


class VehicleCounter(VideoFeatureProcessor):
    def __init__(self):
        super().__init__()
        self.counter_core = VehicleProcessor()

    def get_user_input(self, first_frame: np.ndarray) -> None:
        """Let user select counting line by clicking two points"""
        resized_frame = cv2.resize(
            first_frame, (TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT)
        )
        self.counter_core._select_counting_line(resized_frame)
        self.roi_data = self.counter_core.line_points
        if len(self.roi_data) != 2:
            raise RuntimeError("Counting line not properly selected, exiting")

    def process_frame(
        self,
        frame: np.ndarray,
        current_time: float,
        dt: float,
        detections: Any,
        class_names: dict,
    ) -> np.ndarray:
        """Process frame for vehicle counting and add visualizations"""
        # Process detections and update counts
        self.counter_core._process_detections(detections, class_names)

        # Draw all annotations on the frame
        annotated_frame = self.counter_core._draw_annotations(frame, detections)

        return annotated_frame

    def finalize(self):
        """Save final results to log file"""
        self.counter_core._save_log()

    def print_results(self):
        """Print summary of counting results"""
        self.counter_core.print_summary()

    def get_counts(self):
        """Get dictionary of all counts"""
        return self.counter_core.get_counts()
