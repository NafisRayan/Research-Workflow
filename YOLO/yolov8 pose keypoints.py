import cv2
import ultralytics

model = ultralytics.YOLO(model='yolov8m-pose.pt')


video_path = '2.mp4'  
cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print('dur')
        break

    # Perform pose detection on the current frame
    results = model.predict(frame, save=False)

    # Process each detected person
    for result in results:
        # Extract keypoints
        keypoints = result.keypoints.xyn.cpu().numpy()

        # Get frame dimensions
        height, width, _ = frame.shape

        # Draw keypoints on the frame
        for person_keypoints in keypoints:
            for point in person_keypoints:
                x, y = point
                x = int(x * width)
                y = int(y * height)
                print('point', (x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at each keypoint

    # Display the frame with keypoints
    cv2.imshow('Pose Keypoints', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()