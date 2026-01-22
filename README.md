# Real-Time Vehicle Detection and Traffic Flow Counting System

A web-based application that detects and counts vehicles crossing a defined counting line in video footage. Perfect for traffic analysis, road monitoring, and vehicle flow statistics.

## What It Does

Upload a traffic video, draw a detection area and counting line on the first frame, and the system processes the entire video to track and count vehicles. You get a processed video with visual overlays and an Excel report with detailed statistics.

## Technologies Used

- **Flask** - Web framework for the user interface
- **YOLOv8** - State-of-the-art object detection model
- **OpenCV** - Video processing and computer vision
- **Ultralytics** - YOLO implementation and tracking
- **openpyxl** - Excel report generation

## Features

- Interactive region selection on a web interface
- Real-time processing progress with frame counter
- Tracks 5 vehicle classes: Bus, Car, Motorcycle, Rickshaw, Truck
- Exports counted video with bounding boxes and statistics
- Generates Excel report with count breakdown and percentages

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv env
env\Scripts\activate  # Windows
source env/bin/activate  # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure you have the model file

Place "trained" YOLO model as `best.pt` in the project root directory.

### 4. Run the application

```bash
cd ui
python flask_app.py
```

The app will start at `http://localhost:5000`

## How to Use

1. Upload your traffic video (MP4, AVI, MOV)
2. Click 4 points to define the detection area
3. Click 2 points to define the counting line
4. Start processing and wait for completion
5. Download the processed video and Excel report

## Project Structure

```
yolo_model_ui/
├── best.pt                 # YOLO model weights
├── requirements.txt        # Python dependencies
├── ui/
│   ├── flask_app.py       # Flask backend
│   ├── templates/
│   │   └── index.html     # Web interface
│   ├── uploads/           # Uploaded videos
│   └── outputs/           # Processed videos and reports
└── env/                   # Virtual environment
```

## Notes

Processing time depends on video length and resolution. The system processes each frame for accurate tracking, so longer videos will take more time. Progress is shown in real-time with percentage and frame counts.
