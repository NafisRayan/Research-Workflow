import cv2
import mediapipe as mp 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Minimum confidence thresholds for detection
min_pose_confidence = 0.5
min_tracking_confidence = 0.5

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=min_pose_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make pose detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
       
        # Get the landmarks for left and right wrists
        left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Check if the left hand is raised (left wrist y-coordinate below a certain threshold)
        if left_wrist.y < 0.2:
            cv2.putText(image, 'Left Hand Raised', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Check if the right hand is raised (right wrist y-coordinate below a certain threshold)
        if right_wrist.y < 0.2:
            cv2.putText(image, 'Right Hand Raised', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('MediaPipe Pose', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
            
cap.release()
cv2.destroyAllWindows()
