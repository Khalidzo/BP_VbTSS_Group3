import cv2
import numpy as np
import logging
import time
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class LaneType(Enum):
    """Types of detected lanes"""
    SOLID_WHITE = "solid_white"
    SOLID_YELLOW = "solid_yellow"
    DASHED_WHITE = "dashed_white"
    DASHED_YELLOW = "dashed_yellow"
    DOUBLE_YELLOW = "double_yellow"
    HIGHWAY_EDGE = "highway_edge"

class LaneDetector:
    """Advanced lane detection with automatic classification"""
    
    def __init__(self, warmup_frames=90):  # 3 seconds at 30fps
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.is_warmed_up = False
        
        # Lane detection parameters
        self.roi_vertices = None
        self.lane_history = deque(maxlen=10)
        self.stable_lanes = []
        
        # Detection thresholds
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 30
        self.min_line_length = 40
        self.max_line_gap = 20
        
        # Lane classification parameters
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 30, 255])
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        
        # Colors for different lane types
        self.lane_colors = {
            LaneType.SOLID_WHITE: (255, 255, 255),
            LaneType.SOLID_YELLOW: (0, 255, 255),
            LaneType.DASHED_WHITE: (200, 200, 255),
            LaneType.DASHED_YELLOW: (100, 255, 255),
            LaneType.DOUBLE_YELLOW: (0, 200, 255),
            LaneType.HIGHWAY_EDGE: (255, 100, 100)
        }
        
        logger.info(f"Lane detector initialized. Warmup period: {warmup_frames} frames")
    
    def setup_roi(self, frame_shape):
        """Setup region of interest for lane detection"""
        height, width = frame_shape[:2]
        
        # Define ROI as trapezoid focusing on road area
        roi_vertices = np.array([
            [int(width * 0.1), height],                    # Bottom left
            [int(width * 0.4), int(height * 0.6)],         # Top left
            [int(width * 0.6), int(height * 0.6)],         # Top right
            [int(width * 0.9), height]                     # Bottom right
        ], dtype=np.int32)
        
        self.roi_vertices = roi_vertices
        logger.info("ROI setup completed")
    
    def apply_roi(self, image):
        """Apply region of interest mask"""
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [self.roi_vertices], 255)
        return cv2.bitwise_and(image, mask)
    
    def detect_white_lanes(self, frame):
        """Detect white lane markings"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        return white_mask
    
    def detect_yellow_lanes(self, frame):
        """Detect yellow lane markings"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        return yellow_mask
    
    def classify_lane_type(self, line_segment, color_mask, frame):
        # Classify the type of lane marking
        x1, y1, x2, y2 = line_segment
        
        # Extract region around the line
        line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Create a mask for the line area
        line_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 10)
        
        # Check color intensity along the line
        white_intensity = np.sum(self.detect_white_lanes(frame) & line_mask)
        yellow_intensity = np.sum(self.detect_yellow_lanes(frame) & line_mask)
        
        # Determine if line is dashed or solid based on continuity
        is_dashed = self.is_dashed_line(line_segment, color_mask)
        
        # Classify based on color and pattern
        if yellow_intensity > white_intensity:
            if is_dashed:
                return LaneType.DASHED_YELLOW
            else:
                # Check for double yellow by looking for parallel lines
                if self.check_double_line(line_segment, color_mask):
                    return LaneType.DOUBLE_YELLOW
                return LaneType.SOLID_YELLOW
        else:
            if is_dashed:
                return LaneType.DASHED_WHITE
            else:
                return LaneType.SOLID_WHITE
    
    def is_dashed_line(self, line_segment, mask):
        """Check if a line is dashed by analyzing gaps"""
        x1, y1, x2, y2 = line_segment
        
        # Sample points along the line
        num_samples = 20
        gaps = 0
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                if mask[y, x] == 0:  # Gap detected
                    gaps += 1
        
        # If more than 30% gaps, consider it dashed
        return gaps / num_samples > 0.3
    
    """def check_double_line(self, line_segment, mask):
        # Check if there's a parallel line nearby (double yellow)
        x1, y1, x2, y2 = line_segment
        
        # Calculate perpendicular direction
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return False
        
        perp_x, perp_y = -dy/length, dx/length
        
        # Check for parallel line at various distances
        for offset in [5, 10, 15]:
            offset_x1 = int(x1 + offset * perp_x)
            offset_y1 = int(y1 + offset * perp_y)
            offset_x2 = int(x2 + offset * perp_x)
            offset_y2 = int(y2 + offset * perp_y)
            
            # Sample points along the offset line
            parallel_line_strength = 0
            samples = 10
            
            for i in range(samples):
                t = i / (samples - 1)
                x = int(offset_x1 + t * (offset_x2 - offset_x1))
                y = int(offset_y1 + t * (offset_y2 - offset_y1))
                
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    if mask[y, x] > 0:
                        parallel_line_strength += 1
            
            if parallel_line_strength / samples > 0.5:
                return True
        
        return False
    """
    
    def detect_lanes(self, frame):
        """Main lane detection function"""
        self.frame_count += 1
        
        # Setup ROI on first frame
        if self.roi_vertices is None:
            self.setup_roi(frame.shape)
        
        # Don't draw lanes during warmup period
        if not self.is_warmed_up:
            if self.frame_count >= self.warmup_frames:
                self.is_warmed_up = True
                logger.info("Lane detection warmup completed - starting lane drawing")
            return frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Apply ROI
        roi_edges = self.apply_roi(edges)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Process detected lines
        detected_lanes = []
        if lines is not None:
            # Create color masks for classification
            white_mask = self.detect_white_lanes(frame)
            yellow_mask = self.detect_yellow_lanes(frame)
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # Filter and classify lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # keep only lines that deviate at least ~30° from the horizontal axis
                angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
                if angle < 30 or angle > 150:          # ← reject almost-horizontal lines
                    continue
                
                # Filter by line length
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length < 30:
                    continue
                
                # Classify lane type
                lane_type = self.classify_lane_type(line[0], combined_mask, frame)
                
                detected_lanes.append({
                    'line': line[0],
                    'type': lane_type,
                    'length': length
                })
        
        # Update lane history for stability
        self.lane_history.append(detected_lanes)
        
        # Get stable lanes (lines that appear consistently)
        self.stable_lanes = self.get_stable_lanes()
        
        # Draw lanes on frame
        return self.draw_lanes(frame.copy())
    
    def get_stable_lanes(self):
        """Get lanes that appear consistently across multiple frames"""
        if len(self.lane_history) < 3:
            return []
        
        stable_lanes = []
        
        # Simple stability check - lanes that appear in recent frames
        recent_lanes = []
        for frame_lanes in list(self.lane_history)[-3:]:
            recent_lanes.extend(frame_lanes)
        
        # Group similar lanes
        for lane in recent_lanes:
            if lane['length'] > 50:  # Only consider longer lanes
                stable_lanes.append(lane)
        
        return stable_lanes
    
    def draw_lanes(self, frame):
        """Draw detected lanes with different colors and styles"""
        overlay = frame.copy()
        
        # Draw ROI for reference (semi-transparent)
        if self.roi_vertices is not None:
            cv2.polylines(overlay, [self.roi_vertices], True, (100, 100, 100), 2)
            cv2.fillPoly(overlay, [self.roi_vertices], (50, 50, 50))
            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Draw detected lanes
        for lane in self.stable_lanes:
            x1, y1, x2, y2 = lane['line']
            lane_type = lane['type']
            color = self.lane_colors.get(lane_type, (255, 255, 255))
            
            # Draw different styles based on lane type
            if 'dashed' in lane_type.value:
                self.draw_dashed_line(frame, (x1, y1), (x2, y2), color, 3)
            elif lane_type == LaneType.DOUBLE_YELLOW:
                # Draw two parallel lines
                self.draw_double_line(frame, (x1, y1), (x2, y2), color, 3)
            else:
                cv2.line(frame, (x1, y1), (x2, y2), color, 4)
            
            # Add lane type label
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            label = lane_type.value.replace('_', ' ').title()
            cv2.putText(frame, label, (mid_x - 50, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add detection status
        status_text = f"Lanes Detected: {len(self.stable_lanes)}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def draw_dashed_line(self, img, pt1, pt2, color, thickness):
        """Draw a dashed line"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and direction
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        dx, dy = (x2-x1)/length, (y2-y1)/length
        
        # Draw dashes
        dash_length = 10
        gap_length = 5
        current_length = 0
        
        while current_length < length:
            # Start of dash
            start_x = int(x1 + current_length * dx)
            start_y = int(y1 + current_length * dy)
            
            # End of dash
            end_length = min(current_length + dash_length, length)
            end_x = int(x1 + end_length * dx)
            end_y = int(y1 + end_length * dy)
            
            cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
            
            current_length += dash_length + gap_length
    
    def draw_double_line(self, img, pt1, pt2, color, thickness):
        """Draw a double line for double yellow"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate perpendicular offset
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            return
        
        perp_x, perp_y = -dy/length * 3, dx/length * 3
        
        # Draw two parallel lines
        cv2.line(img, (int(x1 + perp_x), int(y1 + perp_y)), 
                (int(x2 + perp_x), int(y2 + perp_y)), color, thickness)
        cv2.line(img, (int(x1 - perp_x), int(y1 - perp_y)), 
                (int(x2 - perp_x), int(y2 - perp_y)), color, thickness)
    
    def get_detection_info(self):
        """Get current detection information"""
        return {
            'warmed_up': self.is_warmed_up,
            'frame_count': self.frame_count,
            'lanes_detected': len(self.stable_lanes),
            'warmup_progress': min(100, (self.frame_count / self.warmup_frames) * 100)
        }