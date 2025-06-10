import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
import logging
from video_quality_enhancer import VideoQualityEnhancer
from video_source_handler import VideoSourceHandler
from lane_detection import LaneDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedVideoProcessor:
    """Enhanced video processor with lane detection capabilities"""
    
    def __init__(self, source, enhancement_level='medium', buffer_size=2, 
                 enable_weather=True, enable_lane_detection=True):
        self.original_source = source
        self.enhancement_level = enhancement_level
        self.buffer_size = buffer_size
        self.enable_weather = enable_weather
        self.enable_lane_detection = enable_lane_detection
        
        # Initialize components
        self.enhancer = VideoQualityEnhancer(enable_weather_detection=enable_weather)
        self.source_handler = VideoSourceHandler()
        self.lane_detector = LaneDetector() if enable_lane_detection else None
        self.cap = None
        self.source = None
        
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
        
        logger.info(f"Enhanced processor initialized with lane detection: {enable_lane_detection}")
        
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
            backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(self.source, backend)
                    if self.cap.isOpened():
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            logger.info(f"Successfully opened with backend: {backend}")
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
            self.cap = cv2.VideoCapture(self.source)
            
        if not self.cap or not self.cap.isOpened():
            samples = self.source_handler.get_sample_urls()
            logger.error("Failed to open video source. Try these sample URLs:")
            for name, url in samples.items():
                logger.error(f"  {name}: {url}")
            raise Exception(f"Failed to open video source: {self.source}")
        
        # Optimize capture settings
        if isinstance(self.source, str) and self.source.startswith(('http', 'https')):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video source initialized: {self.frame_width}x{self.frame_height} @ {self.original_fps} FPS")
        logger.info(f"Enhancement level: {self.enhancement_level}")
        logger.info(f"Weather detection: {'enabled' if self.enable_weather else 'disabled'}")
        logger.info(f"Lane detection: {'enabled' if self.enable_lane_detection else 'disabled'}")
        
    def capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
                
            try:
                self.frame_queue.put(frame, timeout=0.01)
            except:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, timeout=0.01)
                except:
                    pass
                    
    def enhance_frames(self):
        """Enhance frames in separate thread with lane detection"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Step 1: Apply video quality enhancement
                enhanced_frame = self.enhancer.enhance_frame(frame, self.enhancement_level)
                
                # Step 2: Apply lane detection if enabled
                if self.enable_lane_detection and self.lane_detector:
                    enhanced_frame = self.lane_detector.detect_lanes(enhanced_frame)
                
                # Add enhanced frame to queue
                try:
                    self.enhanced_queue.put(enhanced_frame, timeout=0.01)
                except:
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
        if self.fps_counter % 30 == 0:
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
        
        logger.info("Started enhanced threaded video processing")
        
    def stop_processing(self):
        """Stop video processing"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.enhance_thread:
            self.enhance_thread.join(timeout=1)
            
        if self.cap:
            self.cap.release()
            
        logger.info("Stopped enhanced video processing")
        
    def is_running(self):
        """Check if processor is running"""
        return self.running and (self.capture_thread and self.capture_thread.is_alive())
    
    def get_lane_detection_info(self):
        """Get lane detection information"""
        if self.lane_detector:
            return self.lane_detector.get_detection_info()
        return {'enabled': False}
    
    def toggle_lane_detection(self):
        """Toggle lane detection on/off"""
        if self.lane_detector:
            self.enable_lane_detection = not self.enable_lane_detection
            logger.info(f"Lane detection toggled: {'ON' if self.enable_lane_detection else 'OFF'}")
            return self.enable_lane_detection
        return False
    
    def get_weather_status(self):
        """Get current weather detection status"""
        if hasattr(self.enhancer, 'get_weather_info'):
            return self.enhancer.get_weather_info()
        return {'condition': 'unknown', 'detection_enabled': self.enable_weather}