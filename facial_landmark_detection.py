#!/usr/bin/env python3

import cv2
import mediapipe as mp

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

def main():
    # Set up drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
    closest_face_specs = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)  # Red for closest face
    
    # Initialize video capture (camera)
    cap = cv2.VideoCapture(0)  # Try 0 first, change to 1 if needed
    
    # Set up face mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Convert to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = face_mesh.process(image_rgb)
            
            # Draw landmarks if faces are detected
            if results.multi_face_landmarks:
                # Identify the closest face if multiple faces are detected
                closest_idx = 0
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
                
                # Optionally add text to show which face is closest
                if len(results.multi_face_landmarks) > 1:
                    cv2.putText(image, f"Closest face: {closest_idx+1}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the image (flipped horizontally for selfie view)
            cv2.imshow("Face Mesh Detection", cv2.flip(image, 1))
            
            # Break loop on 'q' key press
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 