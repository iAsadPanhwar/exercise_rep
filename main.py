import os
import cv2 as cv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

file_path = os.path.join('C:\\Users\\asada\\Desktop\\Computer Vision\\Exercise\\Reps\\dumbells.mp4')

if not os.path.exists(file_path):
    print(f"Error: The file path '{file_path}' does not exist.")
    exit()

cap = cv.VideoCapture(0)

counter = 0
stage = None

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Setup MediaPipe instance
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Recolor image to RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor image back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        
        # Extract landmarks
        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv.putText(image, str(angle),
                           tuple(np.multiply(elbow, [image.shape[1], image.shape[0]]).astype(int)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
                
                # Curl counter logic
                if angle > 160:
                    stage = 'down'
                if angle < 30 and stage == 'down':
                    stage = 'up'
                    counter += 1
                    print(f"Rep count: {counter}")
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Warning: Pose landmarks not detected")
        
        # Render curl counter
        # Setup status box
        cv.rectangle(image, (0, 0), (70, 73), (245, 117, 16), -1)
        
        # Rep data
        cv.putText(image, 'Reps', (15, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(image, str(counter), 
                   (10, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
        
        cv.rectangle(image, (2000, 0), (70, 73), (0, 0, 0), -1)
        
        cv.putText(image, 'Arm Position', (575, 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(image, stage, 
                   (550, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
        
        # Render detections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        cv.imshow("Frame", image)
        
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()