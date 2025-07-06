import cv2
import argparse


from denoise import  apply_fast_denoising
from utils import resize_frame, get_video_properties
import sharpen
import denoise
import fog_enhancement
from night_enhancement import NightHighwayEnhancer
import dehazing
import time

class VideoEnhancer:
    def __init__(self, input_path, filters=None, enhancement_type='all', target_width=1280):

        if filters is None:
            filters = []
        self.input_path = input_path
        self.enhancement_type = enhancement_type
        self.target_width = target_width
        self.video_props = get_video_properties(input_path)
        self.window_name = "Enhanced Video Preview"
        self.filters = filters

        # Create display window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.target_width,
                         int(self.video_props['height'] * (self.target_width / self.video_props['width'])))

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
        print(self.filters)
        night_enhancer = NightHighwayEnhancer()

        # Apply selected enhancements
        if 'fog' in self.filters :
            frame = fog_enhancement.remove_fog_contrast(frame)
        if 'night' in self.filters :
            frame = night_enhancer.enhance_frame(frame)
        if 'sharpness' in self.filters :
            frame = sharpen.sharpen_frame(frame)
        if 'snow' in self.filters :
            frame = sharpen.adjust_brightness_contrast(frame,1.1,-30)
            frame = denoise.apply_median_blur(frame)
            frame = dehazing.apply_dehazing(frame,0.1,0.9)
            frame = sharpen.apply_clahe(frame,3.0,(8,8))
        #frame = denoise.apply_bilateral_filter(frame)
        return frame

    def process_video(self):
        """
        Process and display the video frame by frame
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.input_path}")
            return

        frame_count = 0
        start_time = time.time()
        paused = False
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("videos/output_path.mp4", fourcc, fps, (width, height))

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop the video when it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                processed_frame = self.process_frame(frame)
                out.write(processed_frame)
                cv2.imshow(self.window_name, processed_frame)

                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"Processed {frame_count} frames ({fps:.2f} fps)")

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('p'):  # Pause/Unpause
                paused = not paused
            elif key == ord('f'):  # Fast forward
                for _ in range(30):  # Skip 30 frames
                    cap.grab()
                    frame_count += 1
            elif key == ord('1'):  # Toggle fog removal
                if 'night' in self.filters:
                   self.filters.remove('night')
                else:
                    self.filters.append('night')
            elif key == ord('2'):  # Toggle sharpness
                if 'fog' in self.filters:
                    self.filters.remove('fog')
                else:
                    self.filters.append('fog')
            elif key == ord('3'):  # Toggle dust removal
                if 'snow' in self.filters:
                    self.filters.remove('snow')
                else:
                    self.filters.append('snow')
            elif key == ord('4'):
                if 'sharpness' in self.filters:
                    self.filters.remove('sharpness')
                else:
                    self.filters.append('sharpness')
            elif key == ord('a'):  # Toggle all enhancements
                if self.enhancement_type == 'all':
                    self.enhancement_type = 'none'
                else:
                    self.enhancement_type = 'all'

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Highway Video Enhancement - Real-time Display')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--enhancement', choices=['fog', 'sharpness', 'dust', 'all'], default='sharpness',
                        help='Type of enhancement to apply')
    parser.add_argument('--width', type=int, default=1280,
                        help='Target width for display (maintains aspect ratio)')

    args = parser.parse_args()

    enhancer = VideoEnhancer(
        args.input_video,
        [],
        args.enhancement,
        args.width
    )
    enhancer.process_video()