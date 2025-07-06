import cv2
import numpy as np


class FrameZoom:
    def __init__(self):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.zoom_step = 0.1
        self.pan_step = 20

    def apply_zoom(self, frame):
        """Apply zoom and pan to the entire frame"""
        if self.zoom_level == 1.0 and self.pan_x == 0 and self.pan_y == 0:
            return frame

        height, width = frame.shape[:2]

        # Calculate the size of the region to extract
        extract_width = int(width / self.zoom_level)
        extract_height = int(height / self.zoom_level)

        # Calculate center with pan offset
        center_x = width // 2 + self.pan_x
        center_y = height // 2 + self.pan_y

        # Calculate extraction boundaries
        start_x = max(0, center_x - extract_width // 2)
        start_y = max(0, center_y - extract_height // 2)
        end_x = min(width, start_x + extract_width)
        end_y = min(height, start_y + extract_height)

        # Adjust if we hit boundaries
        if end_x - start_x < extract_width:
            start_x = max(0, end_x - extract_width)
        if end_y - start_y < extract_height:
            start_y = max(0, end_y - extract_height)

        # Extract the region
        extracted = frame[start_y:end_y, start_x:end_x]

        # Resize back to original frame size
        zoomed_frame = cv2.resize(extracted, (width, height), interpolation=cv2.INTER_CUBIC)

        return zoomed_frame

    def handle_keyboard(self, key):
        """Handle keyboard input for zoom and pan controls"""
        if key == ord('+') or key == ord('='):  # Zoom in
            self.zoom_level = min(5.0, self.zoom_level + self.zoom_step)
            print(f"Zoom level: {self.zoom_level:.1f}x")
            return True

        elif key == ord('-') or key == ord('_'):  # Zoom out
            self.zoom_level = max(0.5, self.zoom_level - self.zoom_step)
            print(f"Zoom level: {self.zoom_level:.1f}x")
            return True

        elif key == ord('w') or key == ord('W'):  # Pan up
            self.pan_y = max(-200, self.pan_y - self.pan_step)
            print(f"Pan Y: {self.pan_y}")
            return True

        elif key == ord('s') or key == ord('S'):  # Pan down
            self.pan_y = min(200, self.pan_y + self.pan_step)
            print(f"Pan Y: {self.pan_y}")
            return True

        elif key == ord('a') or key == ord('A'):  # Pan left
            self.pan_x = max(-200, self.pan_x - self.pan_step)
            print(f"Pan X: {self.pan_x}")
            return True

        elif key == ord('d') or key == ord('D'):  # Pan right
            self.pan_x = min(200, self.pan_x + self.pan_step)
            print(f"Pan X: {self.pan_x}")
            return True

        elif key == ord('r') or key == ord('R'):  # Reset
            self.zoom_level = 1.0
            self.pan_x = 0
            self.pan_y = 0
            print("Reset to original view")
            return True
        return False

    def draw_info(self, frame):
        """Draw zoom information on the frame"""
        info_frame = frame.copy()

        # Draw zoom level
        zoom_text = f"Zoom: {self.zoom_level:.1f}x"
        cv2.putText(info_frame, zoom_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw pan information
        pan_text = f"Pan: ({self.pan_x}, {self.pan_y})"
        cv2.putText(info_frame, pan_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw controls
        controls = [
            "+/- : Zoom in/out",
            "WASD: Pan",
            "R: Reset",
            "1-5: Quick zoom",
            "ESC: Exit"
        ]

        for i, control in enumerate(controls):
            cv2.putText(info_frame, control, (10, 100 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return info_frame


def process_video_with_zoom(video_source=0, output_callback=None):
    """
    Process video with zoom functionality

    Args:
        video_source: Camera index or video file path
        output_callback: Function to call with zoomed frame for further processing
    """
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    zoom_tool = FrameZoom()

    print("Frame Zoom Tool")
    print("=" * 30)
    print("Controls:")
    print("  +/- : Zoom in/out")
    print("  WASD: Pan around")
    print("  R   : Reset view")
    print("  ESC : Exit")
    print("=" * 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply zoom to the frame
        zoomed_frame = zoom_tool.apply_zoom(frame)

        # Call your existing processing function here
        if output_callback:
            processed_frame = output_callback(zoomed_frame)
        else:
            processed_frame = zoomed_frame

        # Display with zoom info
        display_frame = zoom_tool.draw_info(processed_frame)
        cv2.imshow('Zoomed Frame', display_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif zoom_tool.handle_keyboard(key):
            continue  # Zoom/pan was handled

    cap.release()
    cv2.destroyAllWindows()
