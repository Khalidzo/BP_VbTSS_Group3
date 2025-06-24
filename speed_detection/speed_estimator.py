from base_feature import VideoFeatureProcessor
from speed_detection.byte_tracker import BYTETracker, Track
from speed_detection.visualization import draw_rois, draw_tracks
from speed_detection.roi_processing import (
    process_rois_for_frame,
    cleanup_old_vehicle_data,
)
from speed_detection.utils import format_detections
from speed_detection.roi_selector import ROISelector
from speed_detection.config import PIXELS_PER_METER


def get_rois_from_user(frame, n_rois: int):
    roi_selector = ROISelector(frame, n_rois)
    roi_data = roi_selector.select_rois()
    if roi_data is None:
        raise RuntimeError("ROI selection cancelled by user")
    roi_selector.calculate_perspective_transforms(PIXELS_PER_METER)
    return roi_data


class SpeedEstimator(VideoFeatureProcessor):
    def __init__(self):
        super().__init__()
        self.vehicle_data = {}
        self.tracker = BYTETracker(track_thresh=0.3, track_buffer=30, match_thresh=0.7)

    def get_user_input(self, first_frame):
        n_rois = int(
            input("\nHow many ROIs do you want to select? (Enter an integer)\n> ")
        )
        self.roi_data = get_rois_from_user(first_frame, n_rois)
        self.vehicle_data = {roi["id"]: {} for roi in self.roi_data}

    def process_frame(self, frame, current_time, dt, detections, class_names: dict):
        bboxes, scores, class_ids = format_detections(detections)

        frame = draw_rois(frame, self.roi_data)

        byte_tracks = self.tracker.update(bboxes, scores, class_ids)
        tracks = [Track(t) for t in byte_tracks]

        process_rois_for_frame(
            tracks, self.roi_data, self.vehicle_data, current_time, dt
        )
        draw_tracks(frame, tracks, self.roi_data, self.vehicle_data)
        cleanup_old_vehicle_data(self.vehicle_data, tracks, current_time)

        return frame
