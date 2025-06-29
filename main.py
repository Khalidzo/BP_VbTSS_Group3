from typing import List
import cv2 as cv
import torch
from ultralytics import YOLO
from InquirerPy.prompts.list import ListPrompt
from InquirerPy.prompts.checkbox import CheckboxPrompt
from InquirerPy.prompts.input import InputPrompt
from base_feature import VideoFeatureProcessor
from config import TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT, VEHICLE_CLASSES
from count_vehicles.config import TRACKER_CONFIG
from speed_detection.speed_estimator import SpeedEstimator
from traffic_jam_detection.detect_congestion import CongestionDetector
from count_vehicles.vehicle_counter import VehicleCounter
import tkinter as tk
from tkinter import filedialog
import os
import sys

from utils import draw_label_with_bg

# ──────── Source Selection ────────
source_type = ListPrompt(
    message="Select video source:",
    choices=[
        {"name": "Live Stream (.m3u8)", "value": "live"},
        {"name": "Local Video File", "value": "local"},
    ],
).execute()

# ──────── Video Setup ────────
if source_type == "live":
    stream_url = InputPrompt(message="Paste live stream URL (.m3u8):").execute()
    cap = cv.VideoCapture(stream_url)
    VIDEO_FPS = cap.get(cv.CAP_PROP_FPS) or 30  # Fallback if FPS not known
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("❌ Could not read first frame from live stream")
else:
    # Multiple fallback methods for file selection
    video_path = None

    # Method 1: Try tkinter file dialog with enhanced settings
    try:
        print("📁 Attempting to open file dialog...")

        root = tk.Tk()
        root.withdraw()

        # Make window appear on top and get focus
        root.wm_attributes("-topmost", 1)
        root.update()
        root.lift()
        root.focus_force()

        # Add a small delay to ensure window is ready
        root.after(100)  # type: ignore
        root.update()

        video_path = filedialog.askopenfilename(
            title="Select a Local Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.m4v *.m3u8"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*"),
            ],
            parent=root,
        )

        root.destroy()

        if video_path and video_path.strip():
            print(f"✅ File selected via dialog: {os.path.basename(video_path)}")
        else:
            print("❌ No file selected from dialog")
            video_path = None

    except Exception as e:
        print(f"❌ File dialog failed: {e}")
        video_path = None

    # Method 2: Manual path input if dialog failed
    if not video_path:
        print("\n" + "=" * 50)
        print("📁 FILE SELECTION - Manual Input")
        print("=" * 50)
        print(
            "Since the file dialog didn't work, please provide the video file path manually."
        )
        print("\nOptions:")
        print("1. Type the full path to your video file")
        print("2. Drag and drop the file into this terminal window")
        print("3. Copy the file path from Windows Explorer")
        print("\nExample: C:\\Users\\YourName\\Videos\\video.mp4")
        print("-" * 50)

        while not video_path:
            try:
                user_input = input("\nEnter video file path: ").strip()

                if not user_input:
                    print("❌ Empty path. Please try again.")
                    continue

                # Clean up the path (remove quotes if present)
                video_path = user_input.strip('"').strip("'")

                # Convert forward slashes to backslashes on Windows
                if os.name == "nt":
                    video_path = video_path.replace("/", "\\")

                print(f"🔍 Checking path: {video_path}")
                break

            except KeyboardInterrupt:
                print("\n❌ File selection cancelled by user.")
                sys.exit(1)
            except Exception as e:
                print(f"❌ Error reading input: {e}")
                continue

    # Verify file exists and is readable
    if not os.path.exists(video_path):
        print(f"❌ File does not exist: {video_path}")
        print("   Please check the path and try again.")
        sys.exit(1)

    if not os.path.isfile(video_path):
        print(f"❌ Path is not a file: {video_path}")
        sys.exit(1)

    print(f"✅ Using video file: {os.path.basename(video_path)}")
    print(f"   Full path: {video_path}")

    cap = cv.VideoCapture(video_path)

    # Check if video file opened successfully
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_path}")
        print("   Make sure the file format is supported by OpenCV")
        print("   Supported formats: .mp4, .avi, .mkv, .mov, .m4v")
        sys.exit(1)

    VIDEO_FPS = cap.get(cv.CAP_PROP_FPS)

    # Validate FPS
    if VIDEO_FPS <= 0:
        print("⚠️  Warning: Could not determine video FPS, using default 30 FPS")
        VIDEO_FPS = 30
    else:
        print(f"📊 Video FPS: {VIDEO_FPS}")

    ret, first_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame from video file")
        print("   The video file might be corrupted or in an unsupported format")
        cap.release()
        sys.exit(1)

    # Get video info
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count / VIDEO_FPS if VIDEO_FPS > 0 else 0
    print(f"📹 Video info: {frame_count} frames, {duration:.1f} seconds")

resized_first_frame = cv.resize(
    first_frame, (TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT)
)

# ──────── Feature Selection (Interactive) ────────
feature_choices = [
    {"name": "Speed Detection", "value": "speed"},
    {"name": "Traffic Jam Detection", "value": "congestion"},
    {"name": "Vehicle Counting", "value": "counting"},
]

selected_feature_keys = CheckboxPrompt(
    message="Select features to activate (use space to select):",
    choices=feature_choices,
    instruction="(Use arrow keys to navigate, space to select, enter to confirm)",
).execute()

# ──────── Shared YOLO Model ────────
print("🤖 Loading YOLO model...")
model = YOLO("yolo11n.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


# ──────── Feature Initialization ────────
selected_features: List[VideoFeatureProcessor] = []

if "speed" in selected_feature_keys:
    speed_estimation = SpeedEstimator()
    speed_estimation.get_user_input(resized_first_frame)
    selected_features.append(speed_estimation)

if "congestion" in selected_feature_keys:
    congestion_detection = CongestionDetector(video_fps=VIDEO_FPS)
    congestion_detection.get_user_input(resized_first_frame)
    selected_features.append(congestion_detection)

if "counting" in selected_feature_keys:
    vehicle_counter = VehicleCounter()
    vehicle_counter.get_user_input(resized_first_frame)
    selected_features.append(vehicle_counter)

if not selected_features:
    print("❌ No features selected. Exiting.")
    sys.exit(1)

# ──────── Main Loop ────────
frame_count = 0
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

print("🎬 Starting video processing... (Press ESC to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("📹 End of video reached or could not read frame")
        break

    frame_count += 1
    frame = cv.resize(frame, (TARGET_SCREEN_WIDTH, TARGET_SCREEN_HEIGHT))

    current_time = frame_count / VIDEO_FPS
    dt = 1.0 / VIDEO_FPS

    # Detections
    detections = model(frame, verbose=False)[0]

    # Tracking
    model.track(frame, persist=True, **TRACKER_CONFIG)[0]

    processed_frame = frame.copy()

    # Draw bounding boxes for target vehicle classes
    if detections.boxes is not None:
        for i, box in enumerate(detections.boxes):
            # Get class ID and name
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Check if this detection is one of our target vehicle classes
            if class_name in VEHICLE_CLASSES:
                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Draw bounding box rectangle
                cv.rectangle(processed_frame, (x1, y1), (x2, y2), (140, 0, 0), 2)
                
                # Add class label and confidence using the custom function
                label = f"{class_name}"
                draw_label_with_bg(processed_frame, label, (x1, y1), 
                                 font_scale=0.6, 
                                 text_color=(255, 255, 255), 
                                 bg_color=(140, 0, 0))

    # Process features
    for feature in selected_features:
        processed_frame = feature.process_frame(
            processed_frame, current_time, dt, detections, model.names
        )

    cv.imshow("Vehicle Feature Processor", processed_frame)

    if cv.waitKey(1) & 0xFF == 27:  # ESC key to exit
        print("🛑 Exit requested by user")
        break


# ──────── Cleanup ────────
print("🧹 Cleaning up...")
cap.release()
cv.destroyAllWindows()

# ──────── Final Output ────────
for feature in selected_features:
    if hasattr(feature, "finalize"):
        feature.finalize()
    if hasattr(feature, "print_results"):
        feature.print_results()

print("✅ Processing complete!")
