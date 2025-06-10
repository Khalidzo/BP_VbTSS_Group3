import cv2
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class WeatherCondition(Enum):
    """Detected weather conditions"""
    CLEAR = "clear"
    FOGGY = "foggy"
    RAINY = "rainy"
    SNOWY = "snowy"
    LOW_LIGHT = "low_light"
    OVERCAST = "overcast"

class WeatherEnhancer:
    """Weather-aware video enhancement system"""
    
    def __init__(self):
        self.setup_weather_enhancers()
        self.detection_cache = {}
        self.frame_counter = 0
        self.detection_interval = 30  # Detect weather every 30 frames for performance
        self.current_weather = WeatherCondition.CLEAR
        
    def setup_weather_enhancers(self):
        """Initialize weather-specific enhancement tools"""
        # Fog/Haze removal
        self.fog_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Rain/Snow enhancement
        self.rain_kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]], dtype=np.float32)
        
        # Low light enhancement
        self.lowlight_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        
        # Gamma tables for different conditions
        self.fog_gamma_table = self._build_gamma_table(0.8)
        self.lowlight_gamma_table = self._build_gamma_table(0.6)
        self.overcast_gamma_table = self._build_gamma_table(1.3)
        
    def _build_gamma_table(self, gamma):
        """Build gamma correction lookup table"""
        inv_gamma = 1.0 / gamma
        return np.array([((i / 255.0) ** inv_gamma) * 255 
                        for i in np.arange(0, 256)]).astype("uint8")
    
    def detect_weather_condition(self, frame):
        """Detect weather condition from frame analysis"""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        contrast = std_brightness / (mean_brightness + 1e-6)
        
        # Edge density (sharpness indicator)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Color analysis
        saturation_mean = np.mean(hsv[:, :, 1])
        
        # Weather condition logic
        if mean_brightness < 80:
            return WeatherCondition.LOW_LIGHT
        elif contrast < 0.3 and edge_density < 0.05:
            if saturation_mean < 100:
                return WeatherCondition.FOGGY
            else:
                return WeatherCondition.OVERCAST
        elif self._detect_precipitation(frame, gray):
            if mean_brightness > 120:
                return WeatherCondition.SNOWY
            else:
                return WeatherCondition.RAINY
        elif mean_brightness > 180 and saturation_mean < 80:
            return WeatherCondition.OVERCAST
        else:
            return WeatherCondition.CLEAR
    
    def _detect_precipitation(self, frame, gray):
        """Detect rain/snow patterns"""
        # Look for vertical streaks (rain) or scattered patterns (snow)
        kernel_rain = np.array([[0, 0, 1],
                               [0, 0, 1],
                               [0, 0, 1]], dtype=np.uint8)
        
        kernel_snow = np.array([[1, 0, 1],
                               [0, 1, 0],
                               [1, 0, 1]], dtype=np.uint8)
        
        # Apply morphological operations
        rain_detected = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_rain)
        snow_detected = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_snow)
        
        rain_score = np.sum(rain_detected > 200) / (gray.shape[0] * gray.shape[1])
        snow_score = np.sum(snow_detected > 200) / (gray.shape[0] * gray.shape[1])
        
        return rain_score > 0.001 or snow_score > 0.001
    
    def enhance_foggy_frame(self, frame):
        """Enhance foggy/hazy conditions"""
        # Dark channel prior-inspired enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.fog_clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply gamma correction
        enhanced = cv2.LUT(enhanced, self.fog_gamma_table)
        
        # Increase contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        
        return enhanced
    
    def enhance_rainy_frame(self, frame):
        """Enhance rainy conditions"""
        # Reduce noise and enhance visibility
        denoised = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # Sharpen to counteract rain blur
        sharpened = cv2.filter2D(denoised, -1, self.rain_kernel)
        
        # Enhance contrast
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.fog_clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def enhance_snowy_frame(self, frame):
        """Enhance snowy conditions"""
        # Reduce overexposure from snow reflection
        enhanced = cv2.convertScaleAbs(frame, alpha=0.9, beta=-10)
        
        # Enhance contrast while preserving snow details
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Slight sharpening
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1, 1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def enhance_lowlight_frame(self, frame):
        """Enhance low light conditions"""
        # Aggressive gamma correction
        enhanced = cv2.LUT(frame, self.lowlight_gamma_table)
        
        # CLAHE on all channels
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.lowlight_clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Noise reduction (important for low light)
        enhanced = cv2.bilateralFilter(enhanced, 7, 50, 50)
        
        return enhanced
    
    def enhance_overcast_frame(self, frame):
        """Enhance overcast conditions"""
        # Boost brightness and saturation
        enhanced = cv2.LUT(frame, self.overcast_gamma_table)
        
        # Enhance saturation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Mild contrast enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_weather_aware(self, frame):
        """Apply weather-aware enhancement"""
        self.frame_counter += 1
        
        # Detect weather condition periodically
        if self.frame_counter % self.detection_interval == 0:
            self.current_weather = self.detect_weather_condition(frame)
            logger.info(f"Detected weather condition: {self.current_weather.value}")
        
        # Apply weather-specific enhancement
        if self.current_weather == WeatherCondition.FOGGY:
            return self.enhance_foggy_frame(frame)
        elif self.current_weather == WeatherCondition.RAINY:
            return self.enhance_rainy_frame(frame)
        elif self.current_weather == WeatherCondition.SNOWY:
            return self.enhance_snowy_frame(frame)
        elif self.current_weather == WeatherCondition.LOW_LIGHT:
            return self.enhance_lowlight_frame(frame)
        elif self.current_weather == WeatherCondition.OVERCAST:
            return self.enhance_overcast_frame(frame)
        else:  # CLEAR
            return frame  # No weather-specific enhancement needed
    
    def get_weather_info(self):
        """Get current weather detection info"""
        return {
            'condition': self.current_weather.value,
            'frame_count': self.frame_counter,
            'detection_interval': self.detection_interval
        }