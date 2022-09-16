import sys, os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from teacher.blazebase import resize_pad, denormalize_detections
from teacher.blazepose import BlazePose as tBlazePose
from teacher.blazepose_landmark import BlazePoseLandmark as tBlazePoseLandmark

from student.blazepose import BlazePose as sBlazePose
from student.blazepose_landmark import BlazePoseLandmark as sBlazePoseLandmark
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
            pose_detector.load_weights(weight)
        pose_detector.load_anchors("src/anchors_pose.npy")

        pose_regressor = tBlazePoseLandmark().to(device)
        #pose_regressor.load_weights("src/blazepose_landmark.pth")
        return pose_detector, pose_regressor



def load_images(filenames):
    xfront = np.zeros((len(filenames), 720, 1280, 3), dtype=np.uint8)
    for i, filename in enumerate(tqdm(filenames)):
        img = cv2.imread(filename)
        xfront[i] = cv2.resize(img, (1280, 720))
    return xfront


def test_detect(pose_detector, pose_regressor, frame, device, idx, type="student"):
    frame = np.ascontiguousarray(frame[:,::-1,::-1])
    img1, img2, scale, pad = resize_pad(frame)
    #, img2.shape, scale, pad)
    normalized_pose_detections = pose_detector.predict_on_image(img2)
    pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)
    #print("pose_detections", pose_detections.shape)
    xc, yc, scale, theta = pose_detector.detection2roi(pose_detections)
    img, affine, box = pose_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks, mask = pose_regressor(img.to(device))
    landmarks = pose_regressor.denormalize_landmarks(normalized_landmarks, affine)

    draw_detections(frame, pose_detections)
    draw_roi(frame, box)
    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag>.5:
            draw_landmarks(frame, landmark, POSE_CONNECTIONS, size=2)
            cv2.imwrite(os.path.join("results", str(type)+"{}.png".format(idx)), frame[:,:,::-1])

if __name__=='__main__':
    filenames = ["test1.jpg", "test2.jpg", "test3.jpg"]
    #import glob
    #filenames = glob.glob("GT_frames/WqeOdpBFATc/*.jpg")
    #filenames.sort()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_detector, teacher_regressor = load_blazepose_net(device, teaher=True)
    student_detector, student_regressor = load_blazepose_net(device, teaher=False, weight=None)
    xfront = load_images(filenames)
    for idx in tqdm(range(len(xfront))):
        test_detect(student_detector, student_regressor, xfront[idx], device, idx, type="student")
        test_detect(teacher_detector, teacher_regressor, xfront[idx], device, idx, type="teacher")

