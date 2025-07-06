import cv2
import numpy as np


def resize_frame(frame, target_width=1280):

    height, width = frame.shape[:2]
    ratio = target_width / width
    new_height = int(height * ratio)
    return cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)

def adjust_gamma(frame, gamma=1.0):
    """
    Adjust gamma correction
    Args:
        frame: Input frame
        gamma: Gamma value (1.0 means no change)
    Returns:
        Gamma-adjusted frame
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def get_video_properties(video_path):
    """Get video properties with error handling"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        raise ValueError("Video has invalid dimensions (0x0). File may be corrupted.")

    props = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': width,
        'height': height,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return props