
import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
import argparse
import logging

from weather_condition import WeatherEnhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoQualityEnhancer:
    """Optimized video quality enhancement for real-time processing"""
    
    def __init__(self,  enable_weather_detection=True):
        self.enable_weather_detection = enable_weather_detection
        self.setup_enhancers()
        
        # Initialize weather enhancer if enabled
        if self.enable_weather_detection:
            self.weather_enhancer = WeatherEnhancer()
            logger.info("Weather-aware enhancement enabled")
        
    def setup_enhancers(self):
        """Initialize lightweight enhancement algorithms"""
        # Fast CLAHE with smaller tile size for speed
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        
        # Optimized kernels
        self.sharpen_kernel = np.array([[-0.5, -1, -0.5],
                                       [-1, 6, -1],
                                       [-0.5, -1, -0.5]], dtype=np.float32)
        
        # Pre-computed gamma correction lookup table
        self.gamma_table = self._build_gamma_table(1.2)
        
    def _build_gamma_table(self, gamma):
        """Pre-compute gamma correction lookup table for speed"""
        inv_gamma = 1.0 / gamma
        return np.array([((i / 255.0) ** inv_gamma) * 255 
                        for i in np.arange(0, 256)]).astype("uint8")
    
    def fast_contrast_enhance(self, frame):
        """Fast contrast enhancement using LAB color space"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def fast_denoise(self, frame):
        """Fast noise reduction using bilateral filter with optimized parameters"""
        return cv2.bilateralFilter(frame, 5, 40, 40)
    
    def fast_sharpen(self, frame):
        """Fast sharpening using pre-computed kernel"""
        sharpened = cv2.filter2D(frame, -1, self.sharpen_kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def fast_gamma_correction(self, frame):
        """Fast gamma correction using lookup table"""
        return cv2.LUT(frame, self.gamma_table)
    
    def auto_brightness_contrast(self, frame, clip_hist_percent=1):
        """Automatic brightness and contrast adjustment"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)
        
        # Calculate cumulative distribution from the histogram
        accumulator = [float(hist[0])]
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    def enhance_frame(self, frame, enhancement_level='medium'):
        """
        Apply optimized enhancement based on level with optional weather awareness
        - 'light': Minimal processing for maximum speed
        - 'medium': Balanced quality and speed
        - 'heavy': Maximum quality (slower)
        - 'weather': Weather-aware enhancement (adaptive)
        """
        # Weather-aware enhancement mode
        if enhancement_level == 'weather' and self.enable_weather_detection:
            weather_enhanced = self.weather_enhancer.enhance_weather_aware(frame)
            # Apply additional standard enhancement if needed
            return self.fast_sharpen(weather_enhanced)
        
        # Standard enhancement modes
        elif enhancement_level == 'light':
            # Only gamma correction for speed
            return self.fast_gamma_correction(frame)
            
        elif enhancement_level == 'medium':
            # Balanced enhancement
            enhanced = self.fast_contrast_enhance(frame)
            enhanced = self.fast_sharpen(enhanced)
            return enhanced
            
        else:  # heavy
            # Full enhancement pipeline
            enhanced = self.fast_denoise(frame)
            enhanced = self.auto_brightness_contrast(enhanced)
            enhanced = self.fast_contrast_enhance(enhanced)
            enhanced = self.fast_sharpen(enhanced)
            return enhanced
    
    def get_weather_info(self):
        """Get current weather detection information"""
        if self.enable_weather_detection and hasattr(self, 'weather_enhancer'):
            return self.weather_enhancer.get_weather_info()
        return {'condition': 'weather_detection_disabled'}
    
    def toggle_weather_detection(self):
        """Toggle weather detection on/off"""
        self.enable_weather_detection = not self.enable_weather_detection
        if self.enable_weather_detection and not hasattr(self, 'weather_enhancer'):
            self.weather_enhancer = WeatherEnhancer()
        logger.info(f"Weather detection: {'enabled' if self.enable_weather_detection else 'disabled'}")
        return self.enable_weather_detection