import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Create a video capture object
cap = cv2.VideoCapture('2.mp4')  # Replace with your video file path

# Create a Pose object
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Process each frame in the video
while cap.isOpened():
    success, frame = cap.read()

    # Break the loop if there are no more frames
    if not success:
        break

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation on the frame
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])  # Convert normalized x-coordinate to pixels
            y = int(landmark.y * frame.shape[0])  # Convert normalized y-coordinate to pixels
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle on each landmark

    # Display the frame with pose landmarks
    cv2.imshow('MediaPipe Pose', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()