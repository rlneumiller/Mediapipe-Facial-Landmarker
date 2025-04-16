# Facial Landmark Detection

A Python application that uses MediaPipe and OpenCV to detect facial landmarks and identify the closest face to the camera.

## Features

- Real-time facial landmark detection
- Support for multiple faces (up to 3 by default)
- Automatically identifies which face is closest to the camera
- Highlights the closest face in red

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mediapipe-face-detection.git
   cd mediapipe-face-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with:

```
python facial_landmark_detection.py
```

### Controls

- Press 'q' to quit the application

## How It Works

The application uses MediaPipe's Face Mesh solution to detect up to 3 faces simultaneously. When multiple faces are detected, it determines which face is closest to the camera by measuring the area of each face's bounding box (larger area = closer face).

The closest face is highlighted in red, while other faces are shown in green. 