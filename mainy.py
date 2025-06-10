import cv2
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
        
    def setup_display_window(self):
        """Setup the display window with controls"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
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
        for instruction in instructions:
            logger.info(instruction)
            
    def draw_info_overlay(self, frame, fps, enhancement_level):
        """Draw information overlay on frame including weather info"""
        overlay = frame.copy()
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
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
            
            while True:
                # Get enhanced frame
                enhanced_frame = self.processor.get_enhanced_frame()
                
                if enhanced_frame is not None:
                    frame_count += 1
                    self.processor.update_fps()
                    
                    # Add information overlay
                    if show_info:
                        enhanced_frame = self.draw_info_overlay(
                            enhanced_frame, 
                            self.processor.current_fps,
                            enhancement_level
                        )
                    
                    # Resize frame for display if too large
                    display_frame = enhanced_frame
                    if enhanced_frame.shape[1] > 1280:
                        scale = 1280 / enhanced_frame.shape[1]
                        new_width = int(enhanced_frame.shape[1] * scale)
                        new_height = int(enhanced_frame.shape[0] * scale)
                        display_frame = cv2.resize(enhanced_frame, (new_width, new_height))
                    
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
                        logger.info(f"Frame saved: {filename}")
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
                    else:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
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
    
    args = parser.parse_args()
    
    # Create and run application
    app = Main()
    app.run(
        source=args.source,
        enhancement_level=args.enhancement,
        show_info=not args.no_info,
        enable_weather=not args.no_weather
    )

if __name__ == "__main__":
    main()