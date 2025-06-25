import cv2
import numpy as np
import time
import argparse
import logging
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Main:
    """Main application class with weather enhancement support"""
    
    def __init__(self):
        self.processor = None
        self.window_name = "Enhanced Video Stream"
        self.weather_enabled = True
        self.target_width = 1920
        self.target_height = 1080
        self.resize_method = 'letterbox'  # letterbox, crop, or stretch
        self.interpolation = cv2.INTER_LANCZOS4
        
    def setup_display_window(self):
        """Setup the display window with controls"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Add instructions to window
        instructions = [
            "Controls:",
            "Q - Quit",
            "S - Save current frame", 
            "1 - Light enhancement",
            "2 - Medium enhancement", 
            "3 - Heavy enhancement",
            "4 - Weather-aware enhancement",
            "W - Toggle weather detection",
            "F - Toggle fullscreen",
            "I - Toggle info overlay"
        ]
        
        logger.info("Display window setup complete")
        logger.info(f"Target resolution: {self.target_width}x{self.target_height}")
        logger.info(f"Resize method: {self.resize_method}")
        logger.info(f"Interpolation: {self._get_interpolation_name()}")
        for instruction in instructions:
            logger.info(instruction)
            
    def resize_to_target_resolution(self, frame):
        """Resize frame to target resolution with quality preservation options"""
        if frame is None:
            return None
            
        current_height, current_width = frame.shape[:2]
        
        # If already at target resolution, return as is
        if current_width == self.target_width and current_height == self.target_height:
            return frame
        
        if self.resize_method == 'letterbox':
            return self._letterbox_resize(frame, current_width, current_height)
        elif self.resize_method == 'crop':
            return self._crop_resize(frame, current_width, current_height)
        else:  # stretch
            return self._stretch_resize(frame, current_width, current_height)
    
    def _letterbox_resize(self, frame, current_width, current_height):
        """Resize with letterboxing (black bars) to preserve aspect ratio"""
        # Calculate scaling factors
        scale_x = self.target_width / current_width
        scale_y = self.target_height / current_height
        scale = min(scale_x, scale_y)  # Use smaller scale to fit within bounds
        
        # Calculate new dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # Resize with high-quality interpolation
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=self.interpolation)
        
        # Create black canvas
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Center the resized frame
        offset_x = (self.target_width - new_width) // 2
        offset_y = (self.target_height - new_height) // 2
        canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_frame
        
        self._log_resize_info(current_width, current_height, new_width, new_height, "letterboxed")
        return canvas
    
    def _crop_resize(self, frame, current_width, current_height):
        """Resize by cropping to fill the entire frame (may lose some content)"""
        # Calculate scaling factors
        scale_x = self.target_width / current_width
        scale_y = self.target_height / current_height
        scale = max(scale_x, scale_y)  # Use larger scale to fill entire frame
        
        # Calculate oversized dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # Resize with high-quality interpolation
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=self.interpolation)
        
        # Calculate crop offsets to center the crop
        crop_x = (new_width - self.target_width) // 2
        crop_y = (new_height - self.target_height) // 2
        
        # Crop to target dimensions
        cropped_frame = resized_frame[crop_y:crop_y + self.target_height, 
                                    crop_x:crop_x + self.target_width]
        
        self._log_resize_info(current_width, current_height, self.target_width, self.target_height, "cropped")
        return cropped_frame
    
    def _stretch_resize(self, frame, current_width, current_height):
        """Stretch/compress to exact dimensions (may distort aspect ratio)"""
        resized_frame = cv2.resize(frame, (self.target_width, self.target_height), 
                                 interpolation=self.interpolation)
        
        self._log_resize_info(current_width, current_height, self.target_width, self.target_height, "stretched")
        return resized_frame
    
    def _log_resize_info(self, old_w, old_h, new_w, new_h, method):
        """Log resize information"""
        if not hasattr(self, '_last_logged_resolution') or self._last_logged_resolution != (old_w, old_h):
            logger.info(f"Resized from {old_w}x{old_h} to {new_w}x{new_h} ({method})")
            self._last_logged_resolution = (old_w, old_h)
    def _get_interpolation_name(self):
        """Get human-readable interpolation method name"""
        interpolation_names = {
            cv2.INTER_LANCZOS4: "Lanczos",
            cv2.INTER_CUBIC: "Cubic", 
            cv2.INTER_LINEAR: "Linear",
            cv2.INTER_NEAREST: "Nearest"
        }
        return interpolation_names.get(self.interpolation, "Unknown")
        
    def set_interpolation_method(self, method,frame,enhancement_level):
        """Set interpolation method from string"""
        methods = {
            'lanczos': cv2.INTER_LANCZOS4,
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR, 
            'nearest': cv2.INTER_NEAREST
        }
        self.interpolation = methods.get(method.lower(), cv2.INTER_LANCZOS4)
        """Draw information overlay on frame including weather info"""
        overlay = frame.copy()
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Get weather info if available
        weather_info = {'condition': 'N/A'}
        if self.processor and hasattr(self.processor, 'enhancer'):
            weather_info = self.processor.enhancer.get_weather_info()
        
        # Add text information
        info_text = [
            f"FPS: {fps:.1f}",
            f"Enhancement: {enhancement_level.upper()}",
            f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
            f"Resize Method: {self.resize_method.upper()}",
            f"Interpolation: {self._get_interpolation_name()}",
            f"Weather: {weather_info.get('condition', 'N/A').upper()}",
            f"Weather Detection: {'ON' if self.weather_enabled else 'OFF'}",
            "Press 'Q' to quit"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (15, 30 + i * 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
        return frame
        
    def run(self, source, enhancement_level='medium', show_info=True, enable_weather=True):
        """Main application loop"""
        try:
            # Initialize processor
            self.processor = VideoProcessor(source, enhancement_level, enable_weather=enable_weather)
            self.processor.start_processing()
            self.weather_enabled = enable_weather
            
            # Setup display
            self.setup_display_window()
            fullscreen = False
            frame_count = 0
            
            logger.info(f"Starting video enhancement from: {source}")
            logger.info(f"Weather detection: {'enabled' if enable_weather else 'disabled'}")
            logger.info(f"All frames will be resized to: {self.target_width}x{self.target_height}")
            
            while True:
                # Get enhanced frame
                enhanced_frame = self.processor.get_enhanced_frame()
                
                if enhanced_frame is not None:
                    frame_count += 1
                    self.processor.update_fps()
                    
                    # Resize to target resolution (1920x1080)
                    enhanced_frame = self.resize_to_target_resolution(enhanced_frame)
                    
                    # Add information overlay
                    if show_info:
                        enhanced_frame = self.draw_info_overlay(
                            enhanced_frame, 
                            self.processor.current_fps,
                            enhancement_level
                        )
                    
                    # Display frame (no additional resizing needed since it's already at target resolution)
                    display_frame = enhanced_frame
                    
                    # Optional: Scale down for display if screen resolution is smaller than 1920x1080
                    # Uncomment the following lines if you want to fit the display on smaller screens
                    # if enhanced_frame.shape[1] > 1280:  # If display width > 1280, scale down for viewing
                    #     scale = 1280 / enhanced_frame.shape[1]
                    #     new_width = int(enhanced_frame.shape[1] * scale)
                    #     new_height = int(enhanced_frame.shape[0] * scale)
                    #     display_frame = cv2.resize(enhanced_frame, (new_width, new_height))
                    
                    # Display frame
                    cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord('s'):  # Save frame
                    if enhanced_frame is not None:
                        filename = f"enhanced_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, enhanced_frame)
                        logger.info(f"Frame saved: {filename} (Resolution: {self.target_width}x{self.target_height})")
                elif key == ord('1'):  # Light enhancement
                    enhancement_level = 'light'
                    self.processor.enhancement_level = enhancement_level
                    logger.info("Switched to light enhancement")
                elif key == ord('2'):  # Medium enhancement
                    enhancement_level = 'medium'
                    self.processor.enhancement_level = enhancement_level
                    logger.info("Switched to medium enhancement")
                elif key == ord('3'):  # Heavy enhancement
                    enhancement_level = 'heavy'
                    self.processor.enhancement_level = enhancement_level
                    logger.info("Switched to heavy enhancement")
                elif key == ord('4'):  # Weather-aware enhancement
                    enhancement_level = 'weather'
                    self.processor.enhancement_level = enhancement_level
                    logger.info("Switched to weather-aware enhancement")
                elif key == ord('w'):  # Toggle weather detection
                    if hasattr(self.processor, 'enhancer'):
                        self.weather_enabled = self.processor.enhancer.toggle_weather_detection()
                        logger.info(f"Weather detection toggled: {'ON' if self.weather_enabled else 'OFF'}")
                elif key == ord('i'):  # Toggle info overlay
                    show_info = not show_info
                    logger.info(f"Info overlay: {'ON' if show_info else 'OFF'}")
                elif key == ord('f'):  # Toggle fullscreen
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        time.sleep(0.1)
                    else:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        time.sleep(0.1)
                
                # Check if processor is still running
                if not self.processor.is_running():
                    logger.warning("Video processor stopped")
                    break
                    
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources"""
        if self.processor:
            self.processor.stop_processing()
        cv2.destroyAllWindows()
        logger.info("Application cleanup completed")

def main():
    """Entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Real-Time Video Enhancement System with Weather Awareness')
    parser.add_argument('--source', '-s', required=True,
                       help='Video source (file path, URL, or camera index)')
    parser.add_argument('--enhancement', '-e', choices=['light', 'medium', 'heavy', 'weather'],
                       default='medium', help='Enhancement level')
    parser.add_argument('--no-info', action='store_true',
                       help='Hide information overlay')
    parser.add_argument('--no-weather', action='store_true',
                       help='Disable weather detection')
    parser.add_argument('--buffer-size', type=int, default=2,
                       help='Frame buffer size (lower = less lag)')
    parser.add_argument('--resize-method', choices=['letterbox', 'crop', 'stretch'], 
                       default='letterbox', help='Resize method: letterbox (preserve aspect ratio with black bars), crop (fill frame, may cut content), stretch (distort to fit)')
    parser.add_argument('--interpolation', choices=['lanczos', 'cubic', 'linear', 'nearest'],
                       default='lanczos', help='Interpolation method for resizing (default: lanczos for best quality)')
    
    args = parser.parse_args()
    
    # Create and run application
    app = Main()
    
    # Set resize method and interpolation
    app.resize_method = args.resize_method
    app.set_interpolation_method(args.interpolation)
    
    app.run(
        source=args.source,
        enhancement_level=args.enhancement,
        show_info=not args.no_info,
        enable_weather=not args.no_weather
    )

if __name__ == "__main__":
    main()