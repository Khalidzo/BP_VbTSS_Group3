import cv2
import numpy as np
from typing import Tuple

def sharpen_frame(frame):

    gaussian_blur = cv2.GaussianBlur(frame,(7,7),2)

    sharpened_frame = cv2.addWeighted(frame,1.5,gaussian_blur,-0.5,0)
    #Ansatz 2
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    sharpened2 = cv2.filter2D(frame,-1, kernel)

    kernel2 = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    #return cv2.filter2D(frame, -1, kernel2)
    return sharpened2

def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement.
    Useful for bringing out details in flat-looking scenes.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def adjust_brightness_contrast(frame: np.ndarray, alpha: float = 1.2, beta: int = 20) -> np.ndarray:
    """
    Adjusts the brightness and contrast of an image.
    new_image = alpha * original_image + beta
    alpha (contrast control): 1.0-3.0 typical range.
    beta (brightness control): 0-100 typical range.
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)