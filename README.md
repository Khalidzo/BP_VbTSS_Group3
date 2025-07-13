# 🚗 Vehicle Traffic Surveillance System (VTSS) 

This project combines YOLO object detection, multi-object tracking, and various computer vision techniques to provide real-time traffic insights.

## 🎯 Features Overview

### Core Detection & Analysis Modules
- **🏃 Speed Detection**: Real-time vehicle speed estimation using computer vision and tracking algorithms
- **🚦 Traffic Jam Detection**: Congestion analysis and traffic flow monitoring
- **🔢 Vehicle Counting**: Accurate vehicle counting with directional analysis
- **✨ Quality Enhancement**: Real-time video enhancement for better visibility in various conditions

### Video Enhancement Filters
- **🌃 Night Enhancement**: Improves visibility in low-light conditions
- **☁️ Fog Enhancement**: Clarifies video feed in foggy weather
- **❄️ Snow Enhancement**: Enhances visibility during snow conditions  
- **✒️ Sharpness Enhancement**: Increases image clarity and detail

### Interactive Controls
- **🔍 Zoom & Pan**: Real-time video manipulation with keyboard controls
- **⚙️ Dynamic Filter Toggle**: Enable/disable enhancement filters during playback
- **📊 Real-time Analytics**: Live statistics and detection results

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Webcam or video file access

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd BP_VbTSS_Group3
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify YOLO model:**
   - Ensure `yolo11n.pt` is present in the root directory
   - If missing, it will be automatically downloaded on first run

## 🚀 How to Start

### Running the Application

1. **Start the program:**
   ```bash
   python main.py
   ```

2. **Select Video Source:**
   - **Live Stream**: Choose "Live Stream (.m3u8)" and provide a stream URL / [get one from trafficcamarchive.com](https://trafficcamarchive.com/)
   - **Local Video**: Choose "Local Video File" and select from your computer

### Video Source Options

#### Option 1: Live Stream
- Select "Live Stream (ex: .m3u8)"
- Paste your live stream URL when prompted
- Supports most standard streaming formats

#### Option 2: Local Video File
The system provides multiple ways to select your video:

**Method 1 - File Dialog:**
- A file browser will open automatically
- Navigate and select your video file
- Supported formats: `.mp4`, `.avi`, `.mkv`, `.mov`, `.m4v`

**Method 2 - Manual Path Entry:**
- If the file dialog fails, enter the full path manually
- Example: `C:\Users\YourName\Videos\traffic_video.mp4`
- You can drag-and-drop the file into the terminal

## 🎮 How to Use

### Feature Selection
After selecting your video source, choose which analysis features to activate:

- **☑️ Speed Detection**: Estimates vehicle speeds in real-time
- **☑️ Traffic Jam Detection**: Monitors traffic congestion levels  
- **☑️ Vehicle Counting**: Counts vehicles passing through defined areas

*Use arrow keys to navigate, spacebar to select/deselect, and Enter to confirm.*

### Interactive Controls During Playback

#### Video Enhancement (Press keys during playbook):
- **`1`**: Toggle night enhancement filter
- **`2`**: Toggle fog enhancement filter  
- **`3`**: Toggle snow enhancement filter
- **`4`**: Toggle sharpness enhancement filter

#### Zoom & Pan Controls:
- **`+`**: Zoom out
- **`-`**: Zoom in
- **`W`**: Pan up
- **`S`**: Pan down  
- **`D`**: Pan right
- **`A`**: Pan left
- **`R`**: Reset zoom to original view

#### General Controls:
- **`ESC`**: Exit the application

### Initial Setup (First Time Only)
When you first run a feature, you'll be prompted to:

1. **Speed Detection**: Define speed measurement regions (ROIs)
2. **Traffic Jam Detection**: Set congestion monitoring areas
3. **Vehicle Counting**: Draw counting lines or zones

*Follow the on-screen instructions for each feature setup.*

## 📊 What to Expect

### Real-time Display
- **Bounding Boxes**: Vehicles highlighted with colored rectangles
- **Speed Labels**: Real-time speed readings above detected vehicles
- **Count Statistics**: Running totals of vehicle counts
- **Congestion Indicators**: Visual traffic density feedback

### Console Output
- **📊 Video Info**: Frame count, duration, and FPS details
- **🤖 Model Loading**: YOLO model initialization status
- **✅ Feature Status**: Active features and their configurations
- **📈 Live Statistics**: Real-time analytics and detection metrics

### Performance Expectations
- **GPU**: 30+ FPS with all features enabled
- **CPU Only**: 10-15 FPS (may vary by video resolution)
- **Memory Usage**: ~2-4GB RAM depending on video size and features

### Output Files
- **Counter Logs**: Vehicle counting statistics saved to text files
- **Speed Data**: Speed measurements logged for analysis
- **Jam Reports**: Traffic congestion analysis results

## 🏗️ Project Structure

```
BP_VbTSS_Group3/
├── main.py                    # Main application entry point
├── base_feature.py           # Abstract base class for features
├── config.py                 # Global configuration settings
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── yolo11n.pt               # YOLO model weights
│
├── speed_detection/          # Speed estimation module
├── traffic_jam_detection/    # Congestion analysis module  
├── count_vehicles/          # Vehicle counting module
└── preprocessing/           # Video enhancement filters
```

## ⚡ Performance Tips

1. **Use GPU**: Ensure CUDA is available for optimal performance
2. **Video Resolution**: Lower resolution videos process faster
3. **Feature Selection**: Enable only needed features to improve speed
4. **File Format**: MP4 files generally perform better than other formats

## 🔧 Troubleshooting

### Common Issues
- **File Dialog Issues**: Use manual path entry if the file browser doesn't open
- **Low FPS**: Try reducing video resolution or disabling some features
- **CUDA Errors**: Install proper NVIDIA drivers and CUDA toolkit
- **Video Not Opening**: Verify file format and codec compatibility


---