#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Try to import TensorFlow, but continue if not available
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers.experimental.preprocessing import Rescaling
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
    from tensorflow.keras.losses import categorical_crossentropy
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
    print("TensorFlow successfully imported. Emotion recognition enabled.")
except ImportError:
    print("TensorFlow not available. Emotion recognition will be disabled.")

# Define emotions dictionary with labels and colors
emotions = {
    0: ['Angry', (0, 0, 255), (255, 255, 255)],
    1: ['Disgust', (0, 102, 0), (255, 255, 255)],
    2: ['Fear', (255, 255, 153), (0, 51, 51)],
    3: ['Happy', (153, 0, 153), (255, 255, 255)],
    4: ['Sad', (255, 0, 0), (255, 255, 255)],
    5: ['Surprise', (0, 255, 0), (255, 255, 255)],
    6: ['Neutral', (160, 160, 160), (255, 255, 255)]
}

# Model paths
model_path_1 = os.path.join(os.path.dirname(__file__), 'models', 'vggnet.h5')
model_path_2 = os.path.join(os.path.dirname(__file__), 'models', 'vggnet_up.h5')

# Define input shape and number of classes
input_shape = (48, 48, 1)
num_classes = len(emotions)

# Global variables for models
MODEL_1, MODEL_2 = None, None

if TF_AVAILABLE:
    class VGGNet(Sequential):
        def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
            super().__init__()
            self.add(Rescaling(1./255, input_shape=input_shape))
            self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
            self.add(BatchNormalization())
            self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
            self.add(BatchNormalization())
            self.add(MaxPool2D())
            self.add(Dropout(0.5))

            self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
            self.add(BatchNormalization())
            self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
            self.add(BatchNormalization())
            self.add(MaxPool2D())
            self.add(Dropout(0.4))

            self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
            self.add(BatchNormalization())
            self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
            self.add(BatchNormalization())
            self.add(MaxPool2D())
            self.add(Dropout(0.5))

            self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
            self.add(BatchNormalization())
            self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
            self.add(BatchNormalization())
            self.add(MaxPool2D())
            self.add(Dropout(0.4))

            self.add(Flatten())
            
            self.add(Dense(1024, activation='relu'))
            self.add(Dropout(0.5))
            self.add(Dense(256, activation='relu'))

            self.add(Dense(num_classes, activation='softmax'))

            self.compile(optimizer=Adam(learning_rate=lr),
                        loss=categorical_crossentropy,
                        metrics=['accuracy'])
            
            self.checkpoint_path = checkpoint_path

    # Load the models if they exist
    def load_emotion_models():
        print("Loading emotion recognition models...")
        
        try:
            model_1 = VGGNet(input_shape, num_classes, model_path_1)
            model_1.load_weights(model_1.checkpoint_path)
            
            model_2 = VGGNet(input_shape, num_classes, model_path_2)
            model_2.load_weights(model_2.checkpoint_path)
            
            print("Models loaded successfully.")
            return model_1, model_2
        except Exception as e:
            print(f"Error loading models: {e}")
            return None, None

    # Try to load models if TensorFlow is available
    MODEL_1, MODEL_2 = load_emotion_models()

def find_closest_face(multi_face_landmarks, image_width, image_height):
    """
    Identify which face is closest to the camera based on face area.
    Larger face area indicates closer proximity to the camera.
    """
    max_area = 0
    closest_face_idx = 0
    
    for i, face_landmarks in enumerate(multi_face_landmarks):
        # Get x, y coordinates of all landmarks
        x_coords = [landmark.x * image_width for landmark in face_landmarks.landmark]
        y_coords = [landmark.y * image_height for landmark in face_landmarks.landmark]
        
        # Calculate the area using min/max coordinates as a simple bounding box
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        area = width * height
        
        if area > max_area:
            max_area = area
            closest_face_idx = i
            
    return closest_face_idx

def extract_face_from_landmarks(face_landmarks, image, padding=0.1):
    """
    Extract a face region using facial landmarks with padding.
    Returns the cropped face and its bounding box.
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Extract x,y coordinates from landmarks
    x_coords = [landmark.x * width for landmark in face_landmarks.landmark]
    y_coords = [landmark.y * height for landmark in face_landmarks.landmark]
    
    # Calculate bounding box with padding
    x_min = max(0, int(min(x_coords) - padding * width))
    y_min = max(0, int(min(y_coords) - padding * height))
    x_max = min(width, int(max(x_coords) + padding * width))
    y_max = min(height, int(max(y_coords) + padding * height))
    
    # Ensure we have a valid box
    if x_min >= x_max or y_min >= y_max:
        return None, (0, 0, 0, 0)
    
    # Crop face
    face_crop = image[y_min:y_max, x_min:x_max]
    
    return face_crop, (x_min, y_min, x_max, y_max)

def preprocess_face(face_crop):
    """
    Preprocess the face image for the emotion recognition model:
    - Convert to grayscale
    - Resize to 48x48
    - Prepare for model input
    """
    if face_crop is None or face_crop.size == 0:
        return None
    
    try:
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Resize to 48x48
        resized_face = cv2.resize(gray_face, (48, 48))
        
        # Add channel dimension and expand to batch
        processed_face = np.expand_dims(resized_face, axis=-1)
        processed_face = np.expand_dims(processed_face, axis=0)
        
        return processed_face
    except Exception as e:
        print(f"Error in preprocessing face: {e}")
        return None

def detect_emotion(face_landmarks, image):
    """
    Detect emotion using the VGGNet models.
    Returns the emotion label, color for display, and face bounding box.
    """
    global MODEL_1, MODEL_2, TF_AVAILABLE
    
    # Extract face region for visualization
    face_crop, bbox = extract_face_from_landmarks(face_landmarks, image)
    
    # If TensorFlow is not available or models are not loaded, use traditional detection
    if not TF_AVAILABLE or MODEL_1 is None or MODEL_2 is None:
        # Use traditional detection to infer emotions
        expressions = detect_facial_expressions(face_landmarks, image.shape[1], image.shape[0])
        
        # Map expressions to emotions
        if "Smiling" in expressions or "Happy" in expressions:
            return "Happy", emotions[3][1], bbox  # Happy
        elif "Surprised" in expressions or "Mouth Open" in expressions:
            return "Surprise", emotions[5][1], bbox  # Surprise
        elif "Eyes Closed" in expressions or "Tired/Blinking" in expressions:
            return "Neutral", emotions[6][1], bbox  # Neutral
        else:
            return "Neutral", emotions[6][1], bbox  # Default to Neutral
    
    # Preprocess face
    processed_face = preprocess_face(face_crop)
    
    if processed_face is None:
        return "Unknown", (200, 200, 200), bbox
    
    try:
        # Perform inference
        prediction_1 = MODEL_1.predict(processed_face, verbose=0)
        prediction_2 = MODEL_2.predict(processed_face, verbose=0)
        
        # Combine predictions and get emotion
        combined_prediction = prediction_1 + prediction_2
        emotion_index = np.argmax(combined_prediction)
        
        return emotions[emotion_index][0], emotions[emotion_index][1], bbox
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "Unknown", (200, 200, 200), bbox

def detect_smile(face_landmarks, image_width, image_height):
    """
    Detect if a person is smiling based on mouth corner positions.
    
    Key landmarks:
    - 61, 291: mouth corners
    - 0: middle of upper lip
    - 17: middle of lower lip
    """
    # Get mouth corner landmarks and middle point
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]
    upper_middle = face_landmarks.landmark[0]
    lower_middle = face_landmarks.landmark[17]
    
    # Convert normalized coordinates to pixel coordinates
    left_corner_px = (left_corner.x * image_width, left_corner.y * image_height)
    right_corner_px = (right_corner.x * image_width, right_corner.y * image_height)
    upper_middle_px = (upper_middle.x * image_width, upper_middle.y * image_height)
    lower_middle_px = (lower_middle.x * image_width, lower_middle.y * image_height)
    
    # Calculate middle point between the corners
    mid_corners_y = (left_corner_px[1] + right_corner_px[1]) / 2
    
    # Calculate vertical distance from middle point to mouth middle
    mouth_middle_y = (upper_middle_px[1] + lower_middle_px[1]) / 2
    
    # Smiling typically means corners are higher than middle point
    curvature = mouth_middle_y - mid_corners_y
    
    # Normalize by face width to make it scale-invariant
    face_width = abs(right_corner_px[0] - left_corner_px[0])
    normalized_curvature = curvature / face_width if face_width > 0 else 0
    
    # Threshold for smile detection (positive values indicate upward curve)
    # This threshold needs to be calibrated through testing
    smile_threshold = 0.05
    
    return normalized_curvature > smile_threshold

def detect_open_mouth(face_landmarks, image_width, image_height):
    """
    Detect if mouth is open based on distance between upper and lower lip.
    
    Key landmarks:
    - 13: middle of upper lip
    - 14: middle of lower lip
    - 61, 291: mouth corners for normalizing distance
    """
    # Get lip landmarks
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    left_corner = face_landmarks.landmark[61]
    right_corner = face_landmarks.landmark[291]
    
    # Convert to pixel coordinates
    upper_lip_px = (upper_lip.y * image_height)
    lower_lip_px = (lower_lip.y * image_height)
    
    # Calculate vertical distance between lips
    lip_distance = abs(lower_lip_px - upper_lip_px)
    
    # Normalize by face width
    left_corner_px = (left_corner.x * image_width)
    right_corner_px = (right_corner.x * image_width)
    face_width = abs(right_corner_px - left_corner_px)
    normalized_distance = lip_distance / face_width if face_width > 0 else 0
    
    # Threshold for open mouth detection
    open_mouth_threshold = 0.15
    
    return normalized_distance > open_mouth_threshold

def detect_closed_eyes(face_landmarks, image_width, image_height):
    """
    Detect if eyes are closed using Eye Aspect Ratio (EAR).
    
    Key landmarks (for left eye):
    - 159, 145: outer corners
    - 33, 133: upper and lower lids at center
    """
    # Eye landmarks (left eye)
    left_eye_landmarks = [159, 145, 33, 133]  # Outer corners and upper/lower lids
    
    # Extract coordinates
    left_eye_points = [
        (face_landmarks.landmark[idx].x * image_width, 
         face_landmarks.landmark[idx].y * image_height) 
        for idx in left_eye_landmarks
    ]
    
    # Calculate eye aspect ratio (EAR)
    # EAR = distance between upper and lower lids / distance between outer corners
    vertical_dist = abs(left_eye_points[2][1] - left_eye_points[3][1])
    horizontal_dist = abs(left_eye_points[0][0] - left_eye_points[1][0])
    
    ear = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0
    
    # Threshold for closed eyes (lower EAR means more closed eyes)
    closed_eyes_threshold = 0.15
    
    return ear < closed_eyes_threshold

def detect_facial_expressions(face_landmarks, image_width, image_height):
    """Detect various facial expressions based on landmark positions."""
    expressions = []
    
    # Detect individual expressions
    if detect_smile(face_landmarks, image_width, image_height):
        expressions.append("Smiling")
        
    if detect_open_mouth(face_landmarks, image_width, image_height):
        expressions.append("Mouth Open")
        
    if detect_closed_eyes(face_landmarks, image_width, image_height):
        expressions.append("Eyes Closed")
    
    # Combine expressions for emotion detection
    if "Smiling" in expressions and "Eyes Closed" not in expressions:
        expressions.append("Happy")
    elif "Eyes Closed" in expressions and "Smiling" not in expressions:
        expressions.append("Tired/Blinking")
    elif "Mouth Open" in expressions and "Smiling" not in expressions:
        expressions.append("Surprised")
        
    return expressions

def detect_head_pose(face_landmarks, image):
    """
    Detect head pose using the same approach as in HeadPoseEstimation.py.
    Returns the angles, nose points for direction visualization, and direction text.
    """
    img_h, img_w = image.shape[:2]
    face_3d = []
    face_2d = []
    
    # Extract specific landmarks for pose estimation
    key_landmarks = [33, 263, 1, 61, 291, 199]
    nose_2d, nose_3d = None, None
    
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in key_landmarks:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            
            # Store nose point separately
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
            
            # Get the 2D coordinates
            face_2d.append([x, y])
            
            # Get the 3D coordinates
            face_3d.append([x, y, lm.z])
    
    # Convert to NumPy arrays
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    # Camera matrix
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h / 2],
        [0, focal_length, img_w / 2],
        [0, 0, 1]
    ])
    
    # Distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    
    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
    if not success:
        return None, None, None, None
    
    # Get rotational matrix
    rmat, _ = cv2.Rodrigues(rot_vec)
    
    # Get angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    # Convert to degrees
    x = angles[0] * 360  # pitch
    y = angles[1] * 360  # yaw
    z = angles[2] * 360  # roll
    
    # Determine head direction
    if y < -10:
        text = "Looking Left"
    elif y > 10:
        text = "Looking Right"
    elif x < -10:
        text = "Looking Down"
    elif x > 10:
        text = "Looking Up"
    else:
        text = "Forward"
    
    # Calculate nose direction point for visualization
    nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
    
    return (x, y, z), (p1, p2), text

def main():
    # Set up drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
    closest_face_specs = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Red for closest face
    
    # Initialize video capture (camera)
    cap = cv2.VideoCapture(0)  # Try 0 first, change to 1 if needed
    
    # For MacOS, try to set camera properties
    if not cap.isOpened():
        print("Failed to open camera with index 0, trying index 1...")
        cap = cv2.VideoCapture(1)
    
    # If still not working, try different camera APIs on MacOS
    if not cap.isOpened():
        print("Trying different camera API...")
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use AVFoundation API on macOS
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check camera permissions.")
        return
    
    # Set up face mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
        
        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Failed to read frame from camera")
                    # On MacOS, sometimes we need to retry a few times
                    retry_count = 0
                    while not success and retry_count < 3:
                        success, image = cap.read()
                        retry_count += 1
                        time.sleep(0.1)  # Short delay between retries
                    
                    if not success:
                        print("Failed to read from camera after retries")
                        break
                
                # Convert to RGB for MediaPipe processing
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image
                results = face_mesh.process(image_rgb)
                
                # Variables to store facial expressions and emotions
                expressions = []
                emotion = "Unknown"
                emotion_color = (200, 200, 200)
                emotion_bbox = (0, 0, 0, 0)
                closest_idx = 0
                
                # Variables for head pose
                angles = None
                nose_points = None
                head_direction = None
                
                # Draw landmarks if faces are detected
                if results.multi_face_landmarks:
                    # Identify the closest face if multiple faces are detected
                    if len(results.multi_face_landmarks) > 1:
                        closest_idx = find_closest_face(results.multi_face_landmarks, 
                                                       image.shape[1], image.shape[0])
                    
                    # Draw each face with appropriate styling
                    for i, face_landmarks in enumerate(results.multi_face_landmarks):
                        # Use different color for the closest face
                        contour_specs = closest_face_specs if i == closest_idx else my_drawing_specs
                        
                        # Draw face mesh tesselation
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style()
                        )
                        
                        # Draw face mesh contours
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=contour_specs
                        )
                        
                        # Detect facial expressions, emotions and head pose for the closest face
                        if i == closest_idx:
                            # Get traditional expressions for comparison
                            expressions = detect_facial_expressions(
                                face_landmarks, image.shape[1], image.shape[0]
                            )
                            
                            # Get deep learning-based emotion recognition
                            emotion, emotion_color, emotion_bbox = detect_emotion(face_landmarks, image)
                            
                            # Get head pose estimation
                            angles, nose_points, head_direction = detect_head_pose(face_landmarks, image)
                            
                            # Draw emotion bounding box and label
                            if emotion_bbox[2] > 0:  # Make sure we have a valid box
                                cv2.rectangle(image, 
                                             (emotion_bbox[0], emotion_bbox[1]), 
                                             (emotion_bbox[2], emotion_bbox[3]), 
                                             emotion_color, 2)
                                
                                # Remove text from original image - we'll add it to flipped image instead
                                cv2.rectangle(image, 
                                             (emotion_bbox[0], emotion_bbox[1]-25), 
                                             (emotion_bbox[0] + 100, emotion_bbox[1]), 
                                             emotion_color, -1)
                            
                            # Draw head pose direction line
                            if nose_points is not None:
                                cv2.line(image, nose_points[0], nose_points[1], (255, 0, 0), 3)
                
                # Flip the image horizontally for selfie view AFTER drawing landmarks
                flipped_image = cv2.flip(image, 1)
                
                # Add text annotations to the flipped image AFTER flipping
                if results.multi_face_landmarks:
                    # Add emotion label to the flipped bounding box
                    if emotion_bbox[2] > 0:
                        # Calculate flipped coordinates (x coordinate needs to be flipped)
                        flipped_x = image.shape[1] - emotion_bbox[0] - 100  # Adjust width of label
                        flipped_y = emotion_bbox[1]
                        
                        # Add the emotion text to the flipped image
                        cv2.putText(flipped_image, emotion, 
                                  (flipped_x + 5, flipped_y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display detected expressions on the flipped image
                    y_position = 30
                    
                    # Display traditional expressions for comparison
                    for expression in expressions:
                        cv2.putText(flipped_image, expression, (10, y_position), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        y_position += 30
                    
                    # Display head pose information
                    if head_direction:
                        cv2.putText(flipped_image, f"Head: {head_direction}", (10, y_position), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        y_position += 30
                        
                        if angles:
                            x, y, z = angles
                            cv2.putText(flipped_image, f"x: {np.round(x, 2)}", (10, y_position), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_position += 30
                            cv2.putText(flipped_image, f"y: {np.round(y, 2)}", (10, y_position), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_position += 30
                            cv2.putText(flipped_image, f"z: {np.round(z, 2)}", (10, y_position), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            y_position += 30
                    
                    # Optionally add text to show which face is closest on the flipped image
                    if len(results.multi_face_landmarks) > 1:
                        cv2.putText(flipped_image, f"Closest face: {closest_idx+1}", (10, y_position), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the flipped image with correctly oriented text
                cv2.imshow("Face Mesh Detection", flipped_image)
                
                # Break loop on 'q' key press
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 