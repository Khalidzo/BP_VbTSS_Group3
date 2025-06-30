from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class VideoFeatureProcessor(ABC):
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

    def finalize(self):
        """Optional method to finalize processing (e.g. save results)."""
        pass

    def print_results(self):
        """Optional method to print results or summaries."""
        pass
