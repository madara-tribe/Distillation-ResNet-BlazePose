import numpy as np
import torch
import cv2

from models.teacher.blazebase import resize_pad, denormalize_detections
from models.teacher.blazepose import BlazePose as tBlazePose
from models.teacher.blazepose_landmark import BlazePoseLandmark as tBlazePoseLandmark

from models.student.blazepose import BlazePose as sBlazePose
from models.student.blazepose_landmark import BlazePoseLandmark as sBlazePoseLandmark
from utils.visualization import draw_detections, draw_landmarks, draw_roi, POSE_CONNECTIONS


def load_blazepose_net(device, teaher=True, weight=None):
    if teaher:
        pose_detector = tBlazePose().to(device)
        pose_detector.load_weights("src/blazepose.pth")
        pose_detector.load_anchors("src/anchors_pose.npy")

        pose_regressor = tBlazePoseLandmark().to(device)
        pose_regressor.load_weights("src/blazepose_landmark.pth")
        return pose_detector, pose_regressor
    else:
        pose_detector = sBlazePose().to(device)
        if weight:
            pose_detector.load_state_dict(torch.load(weight, map_location=device))
        pose_detector.load_anchors("src/anchors_pose.npy")

        pose_regressor = tBlazePoseLandmark().to(device)
        pose_regressor.load_weights("src/blazepose_landmark.pth")
        return pose_detector, pose_regressor


CAMERA = None
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
pose_detector, pose_regressor = load_blazepose_net(gpu, teaher=False, weight="ch/student_iter418000.pth")

WINDOW='test'
cv2.namedWindow(WINDOW)
if CAMERA:
    capture = cv2.VideoCapture(0)
else:
    capture = cv2.VideoCapture('utils/human_pose.mp4') 
if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False
while hasFrame:
    frame_ct +=1
    frame = np.ascontiguousarray(frame[:,::-1,::-1])
    img1, img2, scale, pad = resize_pad(frame)
    normalized_pose_detections = pose_detector.predict_on_image(img2)
    pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)

    xc, yc, scale, theta = pose_detector.detection2roi(pose_detections)
    img, affine, box = pose_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks, mask = pose_regressor(img.to(gpu))
    landmarks = pose_regressor.denormalize_landmarks(normalized_landmarks, affine)

    draw_detections(frame, pose_detections)
    draw_roi(frame, box)

    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag>.5:
            draw_landmarks(frame, landmark, POSE_CONNECTIONS, size=2)

    cv2.imshow(WINDOW, frame[:,:,::-1])
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
