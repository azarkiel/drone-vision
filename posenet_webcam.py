import tensorflow as tf
import numpy as np
import cv2

from posenet import PoseNet, detect_pose, draw_pose, draw_keypoints

# itialize posenet from the package
model_path = 'posenet_resnet50float_stride16'
posenet = PoseNet(model_path)
# SET UP WEBCAM
# -------------
cap = cv2.VideoCapture(0)

# Set VideoCaptureProperties 
cap.set(3, 1280)    # width = 1280
cap.set(4, 720)     # height = 720
CAMERA_RESOLUTION_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAMERA_RESOLUTION_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
CENTER_X = CAMERA_RESOLUTION_WIDTH//2
CENTER_Y = CAMERA_RESOLUTION_HEIGHT//2
# MAIN LOOP
# ---------
while True:
    success, img = cap.read()   # read webcam capture
	
    # get keypoints for single pose estimation. it is a list of 17 keypoints
    keypoints = posenet.predict_singlepose(img)
	
    # track nose
    nose_pos = keypoints[0]['position']
    nose_x = nose_pos[0] - CENTER_X
    nose_y = nose_pos[1] - CENTER_Y
    cv2.putText(img,f'x_distance:{nose_x} y_distance:{nose_y}', (0,CENTER_Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # draw keypoints to the original image
    threshold = 0.0
    draw_pose(img, keypoints, threshold=threshold)
    draw_keypoints(img, keypoints, threshold=threshold)
    
    poses = detect_pose(keypoints)
    detected_poses = [pose for pose, detected in poses.items() if detected]
    detected_poses = ' '.join(detected_poses) if detected_poses else 'None'
    
    cv2.putText(img, f'{detected_poses} detected', (0,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	
    cv2.imshow("pose", img)                 # show the image with keypoints
    if cv2.waitKey(1) & 0xFF == ord('q'):   # terminate window when press q
        break

cv2.destroyAllWindows()