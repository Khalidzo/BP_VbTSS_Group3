import cv2
import numpy as np
from typing import Tuple, Optional
import argparse
import preprocessing.denoise as denoise
from preprocessing.sharpen import apply_clahe
from preprocessing.denoise import apply_fast_denoising
from preprocessing.fog_enhancement import remove_fog_contrast

class NightHighwayEnhancer:
    def __init__(self):
        """Initialize the night highway video enhancer for YOLO detection."""
        #self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        #self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.blur_kernel = (15, 15)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def gamma_correction(self, frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
        """Apply gamma correction to brighten dark areas."""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(frame, table)

    def brighten_frame(self, frame: np.ndarray) -> np.ndarray:
        matrix = np.ones(frame.shape,dtype='uint8') * 50 #50 initial
        return cv2.add(frame, matrix)
    def higher_contrast(self, frame):
        matrix = np.ones(frame.shape) * 0.8
        return np.uint8(cv2.multiply(np.float64(frame),matrix))
    def adaptative_threshold(self,frame):
        return cv2.adaptiveThreshold(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main enhancement pipeline for a single frame."""

        # Step 1 : Brighten the frame
        brighten = self.brighten_frame(frame)
        #cv2.imshow("brighten", brighten)

        # Step 2: Noise reduction
        denoised = denoise.apply_median_blur(brighten)
        #cv2.imshow("denoised", denoised)

        # Step 3: Gamma correction for overall brightening
        #gamma_corrected = self.gamma_correction(denoised, gamma=2.0)
        #cv2.imshow("Gamma corrected", gamma_corrected)

        #Step 4: Augment contrast for better quality
        #higher_contrast = self.higher_contrast(denoised)
        #cv2.imshow("higher_contrast", higher_contrast)
        #Step 5: Threshold application
        threshold = self.adaptative_threshold(denoised)
        #cv2.imshow("threshold", threshold)


        return denoised

