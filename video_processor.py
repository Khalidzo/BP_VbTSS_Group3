import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
import argparse
import logging
from video_quality_enhancer import VideoQualityEnhancer
from video_source_handler import VideoSourceHandler

#logging set up
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Threaded video processor for real-time enhancement with weather awareness"""
    
    def __init__(self, source, enhancement_level='medium', buffer_size=2, enable_weather=True):
        self.original_source = source
        self.enhancement_level = enhancement_level
        self.buffer_size = buffer_size
        self.enable_weather = enable_weather
        
        # Initialize components
        self.enhancer = VideoQualityEnhancer(enable_weather_detection=enable_weather)
        self.source_handler = VideoSourceHandler()
        self.cap = None
        self.source = None  # Will be set in initialize_capture
        
        # Threading components
        self.frame_queue = Queue(maxsize=buffer_size)
        self.enhanced_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.enhance_thread = None
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def initialize_capture(self):
        """Initialize video capture with enhanced source handling"""
        logger.info(f"Processing video source: {self.original_source}")
        
        # Get the actual video source
        self.source = self.source_handler.get_video_source(self.original_source)
        
        if self.source is None:
            raise Exception(f"Could not extract video source from: {self.original_source}")
        
        logger.info(f"Using video source: {self.source}")
        
        # Initialize capture with appropriate backend
        if isinstance(self.source, str) and self.source.startswith(('http', 'https')):
            # For HTTP/HTTPS streams, try different backends
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.source, backend)
                    if self.cap.isOpened():
                        # Test if we can read a frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"Successfully opened with backend: {backend}")
                            # Reset to beginning
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            break
                        else:
                            self.cap.release()
                            self.cap = None
                except Exception as e:
                    logger.warning(f"Backend {backend} failed: {e}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
        else:
            # For local files or camera
            self.cap = cv2.VideoCapture(self.source)
            
        if not self.cap or not self.cap.isOpened():
            # Show sample URLs for testing
            samples = self.source_handler.get_sample_urls()
            logger.error("Failed to open video source. Try these sample URLs:")
            for name, url in samples.items():
                logger.error(f"  {name}: {url}")
            raise Exception(f"Failed to open video source: {self.source}")
        
        # Optimize capture settings
        if isinstance(self.source, str) and self.source.startswith(('http', 'https')):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for streams
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS for online videos
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video source initialized: {self.frame_width}x{self.frame_height} @ {self.original_fps} FPS")
        logger.info(f"Enhancement level: {self.enhancement_level}")
        logger.info(f"Weather detection: {'enabled' if self.enable_weather else 'disabled'}")
        
    def capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
                
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put(frame, timeout=0.01)
            except:
                # Drop frame if queue is full (prevents lag)
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put(frame, timeout=0.01)
                except:
                    pass
                    
    def enhance_frames(self):
        """Enhance frames in separate thread"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Enhance frame
                
                enhanced_frame = self.enhancer.enhance_frame(frame, self.enhancement_level)
                
                # Add enhanced frame to queue
                try:
                    self.enhanced_queue.put(enhanced_frame, timeout=0.01)
                except:
                    # Drop frame if queue is full
                    try:
                        self.enhanced_queue.get_nowait()
                        self.enhanced_queue.put(enhanced_frame, timeout=0.01)
                    except:
                        pass
                        
            except Empty:
                continue
                
    def get_enhanced_frame(self):
        """Get the latest enhanced frame"""
        try:
            return self.enhanced_queue.get_nowait()
        except Empty:
            return None
            
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.current_fps = 30 / elapsed
            self.fps_start_time = current_time
            
    def start_processing(self):
        """Start threaded video processing"""
        self.initialize_capture()
        self.running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.enhance_thread = threading.Thread(target=self.enhance_frames)
        
        self.capture_thread.daemon = True
        self.enhance_thread.daemon = True
        
        self.capture_thread.start()
        self.enhance_thread.start()
        
        logger.info("Started threaded video processing")
        
    def stop_processing(self):
        """Stop video processing"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.enhance_thread:
            self.enhance_thread.join(timeout=1)
            
        if self.cap:
            self.cap.release()
            
        logger.info("Stopped video processing")
        
    def is_running(self):
        """Check if processor is running"""
        return self.running and (self.capture_thread and self.capture_thread.is_alive())
    
    def get_weather_status(self):
        """Get current weather detection status"""
        if hasattr(self.enhancer, 'get_weather_info'):
            return self.enhancer.get_weather_info()
        return {'condition': 'unknown', 'detection_enabled': self.enable_weather}
    