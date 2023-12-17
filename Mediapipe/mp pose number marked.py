import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose  
pose = mp_pose.Pose()

cap = cv2.VideoCapture('2.mp4')

# Keypoint names for indexing
keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

while True:
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])  
            
            # Annotate the points with keypoint index
            cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_PLAIN, 
                        2, (0, 255, 0), 2)
            
            # Draw circle
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
    # Display            
    cv2.imshow('MediaPipe Pose', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()