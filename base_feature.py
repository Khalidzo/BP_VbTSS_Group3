# processing/base_feature.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class VideoFeatureProcessor(ABC):
    def __init__(self):
        self.roi_data = None

    @abstractmethod
    def get_user_input(self, first_frame: np.ndarray) -> None:
        """Collect necessary user input (e.g. select ROIs) and store any config needed."""
        pass

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        current_time: float,
        dt: float,
        detections: Any,
        class_names: dict,
    ) -> np.ndarray:
        """Processes a frame and returns the modified one."""
        pass
