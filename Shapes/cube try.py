import cv2
import numpy as np

# Create the cube object
vertices = np.array([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,1], [0,1,1], [1,1,1], [1,0,1]])
edges = np.array([[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]])

# Function to draw cube from OpenCV frame
def draw_cube(img, pts):
    img = cv2.polylines(img, [pts[0:4]], True, (0,255,255), 3)
    img = cv2.polylines(img, [pts[4:8]], True, (0,255,255), 3)
    img = cv2.polylines(img, [pts[0:4]+pts[4:8]], True, (0,255,255), 3)
    img = cv2.polylines(img, [pts[8:12]], True, (0,255,255), 3)
    img = cv2.polylines(img, [pts[12:16]], True, (0,255,255), 3)
    img = cv2.polylines(img, [pts[8:12]+pts[12:16]], True, (0,255,255), 3)
    return img

# Function to solve PnP problem
def find_pose(objp, corners2D):
    ret, rvec, tvec = cv2.solvePnP(objp, corners2D, mtx, dist)
    return rvec, tvec

# Initialize the camera
cam = cv2.VideoCapture(0)
ret, frame = cam.read()

# Calibrate the camera
mtx = 100
dist = np.array([[0,0,0,0,0]])
h, w = frame.shape[:2]
corners2D = np.zeros((9,6,2), dtype=np.float32)
corners2D = np.mgrid[0:9,0:6].T.reshape(-1,2)

objp = np.zeros((9*6,3), dtype=np.float32)
objp[:,:2] = corners2D.reshape(-1,2)

# Capture a new frame
ret, frame = cam.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

# Draw cube
if ret == True:
    cv2.drawChessboardCorners(frame, (9,6), corners, ret)
    rvec, tvec = find_pose(objp, corners)
    corners3D = cv2.projectPoints(vertices, rvec, tvec, mtx, dist)
    img = draw_cube(frame, np.int32(corners3D.reshape(-1,2)))
    cv2.imshow('images', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

cam.release()