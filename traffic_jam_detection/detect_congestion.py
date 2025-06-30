import numpy as np
import cv2
from base_feature import VideoFeatureProcessor
from config import TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT
from traffic_jam_detection.traffic_jam_detector import TrafficJamDetector
from typing import Any


class CongestionDetector(VideoFeatureProcessor):
    def __init__(self, video_fps):
        super().__init__()
        self.detector = TrafficJamDetector()
        self.frame_count = 0
        self.last_frame_idx = 0
        self.video_fps = video_fps

    def get_user_input(self, first_frame: np.ndarray) -> None:
        resized_frame = cv2.resize(
            first_frame, (TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT)
        )
        self.detector._select_rois(resized_frame)
        self.roi_data = self.detector.polygons
        if not self.roi_data:
            raise RuntimeError("No ROIs selected, exiting")

    def process_frame(
            self,
            frame: np.ndarray,
            current_time: float,
            dt: float,
            detections: Any,
            class_names: dict,
    ) -> np.ndarray:
        self.frame_count += 1

        # Simplified processing - combines tracking and jam detection in one call
        tracked_objects = self.detector._update_tracking_and_detect_jams(
            detections.boxes, class_names, self.frame_count
        )

        # Draw visualization
        self.detector._draw_visualization(frame, tracked_objects)

        return frame

    def finalize(self):
        self.last_frame_idx = self.frame_count
        # Use the simplified finalize method
        self.detector.finalize_jams(self.last_frame_idx)
        # Save Evaluation
        self.detector.save_jam_evaluation(self.video_fps)

    def print_results(self):
        print("\n--- Traffic Jam Detection Results ---")
        for zone_idx, intervals in enumerate(self.detector.jam_intervals):
            print(f"Zone {zone_idx + 1}:")
            if not intervals:
                print("  No jams detected.")
            else:
                for start_f, end_f in intervals:
                    start_s = round(start_f / self.video_fps, 2)
                    end_s = round(end_f / self.video_fps, 2)
                    print(
                        f"  Jam from {start_s}s to {end_s}s (Frames: {start_f}-{end_f})"
                    )
        print("\n--- Traffic Jam Detection Finished ---")