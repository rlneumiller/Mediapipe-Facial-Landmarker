# Facial Landmark Detection with Emotion Recognition

This script provides real-time facial landmark detection and emotion recognition capabilities using MediaPipe and TensorFlow.

## Features

- **Facial Landmark Detection**: Detects and visualizes 468 facial landmarks in real-time using MediaPipe.
- **Emotion Recognition**: Recognizes 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using a VGGNet deep learning model.
- **Multiple Face Handling**: Detects multiple faces and identifies the closest one based on face area.
- **Fallback Mode**: Uses rule-based expression detection if TensorFlow is not available.

## Requirements

### Basic Requirements
- Python 3.x
- OpenCV
- MediaPipe
- NumPy

### For Advanced Emotion Recognition
- TensorFlow 2.x

## Installation

1. Install the basic requirements:
   ```
   pip install opencv-python mediapipe numpy
   ```

2. (Optional) For advanced emotion recognition, install TensorFlow:
   ```
   pip install tensorflow
   ```

## Usage

Run the script directly:

```
python facial_landmark_detection.py
```

- Press 'q' to quit the application.

## Emotion Recognition Models

The script uses two pre-trained VGGNet models for emotion recognition, which should be placed in the `models` directory:

- `models/vggnet.h5`: Base VGGNet model
- `models/vggnet_up.h5`: Improved VGGNet model

If TensorFlow is not available or the models are not found, the script will fall back to rule-based expression detection.

## Customization

You can modify the following aspects of the script:

- Change detection thresholds in the expression detection functions
- Adjust the face padding in the `extract_face_from_landmarks` function
- Modify the visualization styles in the `main` function

## How It Works

1. The script captures video from the default camera
2. Each frame is processed to detect facial landmarks
3. For the closest face, both rule-based expression detection and deep learning-based emotion recognition are applied
4. Results are visualized on the frame with colored bounding boxes and text labels
5. The frame is flipped horizontally for a selfie-view display

## Emotions Detected

- Angry (Red)
- Disgust (Dark Green)
- Fear (Light Yellow)
- Happy (Purple)
- Sad (Blue)
- Surprise (Green)
- Neutral (Gray) 