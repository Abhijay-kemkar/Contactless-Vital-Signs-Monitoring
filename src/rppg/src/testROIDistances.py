#!/usr/bin/env python3

from rospy_tutorials.msg import Floats
from rppg.msg import RGB
from rospy.numpy_msg import numpy_msg

def cropImage(image, bbox):

    x1 = int(bbox[0])
    x2 = int(bbox[2])
    y1 = int(bbox[1])
    y2 = int(bbox[3])

    #change parameters if they were passed in the wrong order
    if x1 > x2: x1, x2 = copy.copy(x2), copy.copy(x1)
    if y1 < y2: y1, y2 = copy.copy(y2), copy.copy(y1)
    #bot to top, and right to left
    crop_image = image[y2:y1, x1:x2]
    return crop_image

def getROICoordinates (bbox, landmarks, roi_area="forehead"):
    roi_points = [-1, -1, -1, -1]
    if roi_area == "forehead":
        y_lower = [19, 24]
        x_lower = 19
        x_upper = 25
        y_lm_max = float('inf')

    if roi_area == "skin":
        y_lower = [30]
        x_lower = 2
        x_upper = 16
        y_lm_max = float('inf')

    #  ROI coordinates
    for index, (x_lm, y_lm) in enumerate(landmarks):
        # lower bound x line
        if index == x_lower:
            roi_points[0] = x_lm

        # lower y bound
        #if index == y_lower[0] or index == y_lower[1]:
        if index in y_lower:
            if y_lm_max > y_lm:
                y_lm_max = y_lm
                roi_points[1] = y_lm_max

        # upper bound x line
        if index == x_upper:
            #pdb.set_trace()
            roi_points[2] = x_lm

    # upper y bound
    roi_points[3] = bbox[1]
    return roi_points

import pdb
from scripts.config import *
from scripts.FaceDetection import Detector
from os import listdir
from os.path import isfile, join
from imutils import face_utils
import torch
import dlib
import cv2

face_detector = Detector(network_backbone="mobile", face_detection_resolution = FACE_DETECTION_RESOLUTION, cpu=True)
face_detector.loadToDevice(torch.device("cpu"), toTensorRT=False)
landmark_detector = dlib.shape_predictor(TRAINED_MODEL_LANDMARKS)

def callback(msg):
	frame = msg.m_rgb
	bbox = face_detector.forward(frame)

	if len(bbox) > 0:
		for bs in bbox:
			if bs[4] < VIS_THRESH:
				continue

			rectangle = dlib.rectangle(left=bs[0], top=bs[1], right=bs[2], bottom=bs[3])
			landmarks_shape = landmark_detector(frame, rectangle)
			landmarks = face_utils.shape_to_np(landmarks_shape)

			# get forehead ROI points
			roi_coord = getROICoordinates(bs, landmarks)
			roi = cropImage(frame, roi_coord)
			#print(file, "roi shape",roi.shape)
			
			pub.publish(roi.shape)
            
def LandmarkDetector() :

	rospy.init_node('LandmarkROI')

    	pub = rospy.Publisher('roicoordinates', numpy_msg(RGB), queue_size=10)
    
	sub = rospy.Subscriber('image', numpy_msg(RGB), callback)

	rospy.spin()

if __name__ == '__main__':
	LandmarkDetector()

