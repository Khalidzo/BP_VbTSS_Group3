import cv2
import argparse


from preprocessing.utils import resize_frame, get_video_properties
import preprocessing.sharpen as sharpen
import preprocessing.denoise as denoise
import preprocessing.fog_enhancement as fog_enhancement
from preprocessing.night_enhancement import NightHighwayEnhancer
import preprocessing.dehazing as dehazing
import time

class VideoEnhancer:
    def __init__(self, input_path, filters=None, target_width=1280):

        if filters is None:
            filters = []
        self.input_path = input_path
        self.target_width = target_width
        self.video_props = get_video_properties(input_path)
        self.filters = filters



    def process_frame(self, frame):
        """
        Process a single frame based on enhancement type
        Args:
            frame: Input frame
        Returns:
            Enhanced frame
        """
        # Resize frame first
        frame = resize_frame(frame, self.target_width)
        #print(self.filters)
        night_enhancer = NightHighwayEnhancer()

        # Apply selected enhancements
        if 'sharpness' in self.filters :
            frame = sharpen.sharpen_frame(frame)
        if 'fog' in self.filters :
            frame = fog_enhancement.remove_fog_contrast(frame)
        if 'night' in self.filters :
            frame = night_enhancer.enhance_frame(frame)
        if 'snow' in self.filters :
            frame = sharpen.adjust_brightness_contrast(frame,1.1,-30)
            frame = denoise.apply_median_blur(frame)
            frame = dehazing.apply_dehazing(frame,0.1,0.9)
            frame = sharpen.apply_clahe(frame,3.0,(8,8))

        return frame

