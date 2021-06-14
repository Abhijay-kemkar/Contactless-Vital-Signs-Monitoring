import jetson.utils
import time
import cv2
import sys
import torch
import pdb
import dlib
import numpy as np
from config import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import json
from scipy.signal import butter, lfilter
import json
from collections import Counter
import os
import csv
import pandas
import scipy
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        # first face detection time usually very high
        if not self.calls == 0:
            self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls

        return {'diff': self.diff, 'avg': self.avg}

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def getAvg(self):
        if self.avg <= 0.001:
            return 0.001

        else: return self.avg



# adapted from: https://medium.com/analytics-vidhya/real-time-head-pose-estimation-with-opencv-and-dlib-e8dc10d62078
# to estimate pose of face
class PoseEstimator(object):

    def __init__(self, landmarks, frame_shape, focal):
        height, width, channels = frame_shape
        focal_length = focal * width

        model_points = self._ref3DModel()
        ref_image_points = self._ref2dImagePoints(landmarks)
        camera_matrix = self._cameraMatrix(focal_length, (height / 2, width / 2))

        mdists = np.zeros((4, 1), dtype=np.float64)
        self.success, self.rotation_vector, self.translation_vector = cv2.solvePnP(
            model_points, ref_image_points, camera_matrix, mdists)

    def getEulerAngles(self):
        # calculating angle
        rmat, jac = cv2.Rodrigues(self.rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        return angles[0], angles[1], angles[2]

    def _ref3DModel(self):
        modelPoints = [[0.0, 0.0, 0.0],
                       [0.0, -330.0, -65.0],
                       [-225.0, 170.0, -135.0],
                       [225.0, 170.0, -135.0],
                       [-150.0, -150.0, -125.0],
                       [150.0, -150.0, -125.0]]
        return np.array(modelPoints, dtype=np.float64)

    def _ref2dImagePoints(self, landmarks):
        imagePoints = [[landmarks.part(30).x, landmarks.part(30).y],
                       [landmarks.part(8).x, landmarks.part(8).y],
                       [landmarks.part(36).x, landmarks.part(36).y],
                       [landmarks.part(45).x, landmarks.part(45).y],
                       [landmarks.part(48).x, landmarks.part(48).y],
                       [landmarks.part(54).x, landmarks.part(54).y]]
        return np.array(imagePoints, dtype=np.float64)

    # scew parameters estiamted by 1
    def _cameraMatrix(self, focal_length, center):
        cameraMatrix = [[focal_length, 1, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]]
        return np.array(cameraMatrix, dtype=np.float)

def getDirToVideos (data_path, subjects, exclude_trial = []):
    # get all file names
    dir_subjects = []
    exclude_trial = ["/"+ str(x) for x in exclude_trial]
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        # checks if filenames contains a valid video extension and is in subjects list
        try:
            subject_nr = int(dirpath.split("subject")[-1].split('/')[0])
        except:
            subject_nr = -1
        # only adds deepest directories to dir list
        if not dirpath == data_path and subject_nr in subjects and dirnames == []:
            #exclude trials
            if not dirpath[-2:] in exclude_trial:
                dir_subjects.append(dirpath + "/")

    return dir_subjects

#TODO: change this from [lower x, upper y, upper x, lower y]
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

# methods: hsv, conaire, none
def getSkinSegmentationMask(img, method='pitas'):
    if method == 'none':
        return img

    # as mentioned in
    # Non-Contact Heart Rate Detection When Face Information Is Missing during Online Learning
    # Zheng et al, 2020
    elif method == 'pitas':
        h_bounds_low = [0, 25]
        h_bounds_high = [335, 360]
        s_bounds = [51, 153]
        v_bounds = [102, float('inf')]

    # as used in
    # Optimizing Remote Photoplethysmography Using Adaptive Skin Segmentation for Real-Time Heart Rate Monitoring
    # Foouad et al, 2019
    # based on
    # Detector adaptation by maximising agreement between independent data sources
    # Conaire et al, 2007
    elif method == 'conaire':
        h_bounds = [78, 159]
        s_bounds = [60, 255]
        v_bounds = [3,139]

    # convert frame from rgb space to hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # convert from 180° to 360° range
    h = img_hsv[:, :, 0] * 2
    s, v = img_hsv[:, :, 1], img_hsv[:, :, 2]

    # conditions 1
    if method== "conaire":
        mask_h = (h_bounds[0] <= h) * (h <= h_bounds[1])

    elif method == "pitas":
        mask_h = (h_bounds_low[0] <= h) * (h <= h_bounds_low[1]) + (h_bounds_high[0] <= h) * (h <= h_bounds_high[1])
        mask_h[mask_h > 0] = 1

    # conditions 2
    mask_s = (s_bounds[0] <= s) * (s <= s_bounds[1])
    mask_v = (v_bounds[0] <= v) * (v <= v_bounds[1])
    mask_sv = mask_s + mask_v
    mask_sv[mask_sv > 1] = 1

    # combine conditiions
    result = mask_h * mask_sv

    return result


def resolutionToShape(resolution):
    if resolution == 'blackfly':
        return (4096, 3000)
    elif resolution == 'openmv':
        return (2592, 1944)
    elif resolution == '2.7k':
        return (2704, 1520)
    elif resolution == '1080p':
        return (1920, 1080)
    elif resolution == '720p':
        return (1280, 720)
    elif resolution == '480p':
        return (852, 480)
    elif resolution == 'screen':
        display = jetson.utils.glDisplay()
        return (display.GetWidth(), display.GetHeight())
    elif resolution == 'ubfc':
        return (640, 480)
    elif resolution == "bwh_small":
        return (2048,1500)
    elif not resolution:
        return resolution
    else:
        resolution = tuple(map(int, resolution.split(',')))
        #print('Unknown input {:}! Choose from: [blackfly, openmv, screen, 2.7k, 1080p, 720p, 480p, "width,height"] '.format(resolution))
        return resolution


# rotate bounding face box and get angle
# from: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
def getRollAngle(shape):
    # extract the left and right eye (x, y)-coordinates for dlib facedetector
    (lStart, lEnd) = (37, 43)
    (rStart, rEnd) = (43, 48)

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    return angle

def scaleBoundingBox(bbox, factor):
    # Get bounding box
    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]

    # get center of rectangle
    cx, cy = x + w / 2, y + h / 2
    # get side length of rectangle
    f = factor / 2

    # get new coordinates for bounding box
    x1, y1 = int(cx + w*f), int(cy + h*f)
    x2, y2 = int(cx - w*f), int(cy - h*f)

    return [int(x1), int(y1), int(x2), int(y2)]


# crops image along bbox
def cropImage(image, bbox):

    x1 = int(bbox[0])
    x2 = int(bbox[2])
    y1 = int(bbox[1])
    y2 = int(bbox[3])
    #bot to top, and right to left
    crop_image = image[y2:y1, x1:x2]
    return crop_image


# rotate each point of rectangle according to angle
def rotateRectangle(bbox, angle):
    if angle < 0: angle = angle - 360
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    p1 = _rotatePoint([left, bottom], angle)
    p2 = _rotatePoint([right, bottom], angle)
    p3 = _rotatePoint([right, top], angle)
    p4 = _rotatePoint([left, top], angle)

    return (p1, p2, p3, p4)


def rescaleBboxToOriginalFrame(bbox, resolution, cropped_frame_coord, cropped_frame_resolution, resize):
    rescale_x_axis = resolution[0] / cropped_frame_resolution[0]
    rescale_y_axis = resolution[1] / cropped_frame_resolution[1]
    rescale_x_axis = 1 / resize
    rescale_y_axis = 1 / resize

    bbox = boundBbox(bbox, resolution)

    bbox[0], bbox[2] = (bbox[0] + cropped_frame_coord[0]) * rescale_x_axis, (
            cropped_frame_coord[1] + bbox[2]) * rescale_x_axis
    bbox[1], bbox[3] = (bbox[1] + cropped_frame_coord[0]) * rescale_y_axis, (
            cropped_frame_coord[1] + bbox[3]) * rescale_y_axis

    return bbox


# bounds bbounding to 0 and max resolution of frame
def boundBbox(bbox, resolution):
    for i, b in enumerate(bbox):
        if b < 0: bbox[i] = 0
    # remove coordinates larger then frame
    if bbox[0] > resolution[0]: bbox[0] = resolution[0]
    if bbox[1] > resolution[1]: bbox[1] = resolution[1]
    if bbox[2] > resolution[0]: bbox[2] = resolution[0]
    if bbox[3] > resolution[1]: bbox[3] = resolution[1]

    return bbox


# rotates point in coordinate space
def _rotatePoint(coord, angle):
    angle = np.radians(angle)
    rotationMatrix = np.array(((np.cos(angle), -np.sin(angle)),
                               (np.sin(angle), np.cos(angle))))

    result = rotationMatrix.dot(coord)

    return result

# rotate bounding box with angle
def rotateBbox (rectangle, angle):
    center = (rectangle.center().x, rectangle.center().y)
    rect = (center, (rectangle.width(), rectangle.height()), angle)
    rotated_box = cv2.boxPoints(rect)
    rotated_box = np.int0(rotated_box)
    return rotated_box

def estimateHR (signal, fs, returnFFT = False, filter=True):
    if filter:
        signal = butter_bandpass(signal, 50/60, 150/60, fs)

    fft_rppg = np.real(fft(signal))
    freq = np.linspace(0, fs, num=len(fft_rppg))

    freq = freq[0:len(fft_rppg) // 2]
    fft_rppg = fft_rppg[0:len(fft_rppg) // 2]

    hr = int(freq[np.argmax(fft_rppg)] * 60)

    if not returnFFT:
        return hr
    else:
        return hr, freq, fft_rppg

def butter_bandpass(signal, lowcut, highcut, fs, order=5, btype = 'band'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if btype == "low": freq = low
    else: freq = [low, high]

    b, a = butter(order, freq , btype=btype)

    y = lfilter(b, a, signal)
    return y

def calculateRmse (predictions, targets):
    predictions = np.array(predictions, dtype = 'float32')
    targets =  np.array(targets, dtype = 'float32')

    return np.sqrt((( predictions -targets) ** 2).mean())

def calculateMae (predictions, targets):
    predictions = np.array(predictions, dtype='float32')
    targets = np.array(targets, dtype='float32')

    return np.abs(predictions-targets).mean()

def evaluateDict (result):
    gt = result['ground_truth']
    pred = result['prediction']

    mae = calculateMae (pred, gt)
    rmse  = calculateRmse (pred, gt)

    result.update({'mae': mae, 'rmse': rmse})

    return result

def convertToDict (gt, pred, buffer_size ,roi_slices, stride, max_rate_of_change_hr, subject = 0):
    result = {'buffer_size': buffer_size, 'roi_slices': roi_slices,
              'subject': subject , 'ground_truth': gt, 'prediction': pred, 'stride':stride, 'max_rof_hr': max_rate_of_change_hr}

    return result


def findMostCommon( lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def load_ground_truth_values(dir, dataset, trial, cam_name ="flir"):

    if dataset == "ubfc":
        # read gt and convert string to flaots
        ground_truth_file = open(dir + 'ground_truth.txt', "r")
        ground_truth_str = ground_truth_file.read().split()
        ground_truth = [float(i) for i in ground_truth_str]

        # split ground truth in signal and hr
        gt_length = len(ground_truth) // 3
        ground_truth_ppg = ground_truth[:gt_length]
        ground_truth_hr = ground_truth[gt_length:2 * gt_length]

        return ground_truth_hr, None, None , ground_truth_ppg

    elif dataset=="bwh_small":
        subject_nr = dir.split("subject")[-1].split('/')[0]

        #load ppg signal
        data = pandas.read_csv(dir + '/ppg.csv', "r", delimiter=",")
        ground_truth_hr = data["hr"].tolist()
        ground_truth_spo2 = data["spo2"].tolist()
        ground_truth_time_stamps = data["time_stamp"].tolist()

        #preprocess ppg data, lin & nn interpolation
        #
        #nearest neighbor interpolation for beginning and end  of data
        ground_truth_hr = nn_interp_tails(ground_truth_hr)
        ground_truth_hr = nn_interp_tails(ground_truth_hr, reverse=True)
        ground_truth_spo2 = nn_interp_tails(ground_truth_spo2)
        ground_truth_spo2 = nn_interp_tails(ground_truth_spo2, reverse=True)
        #linear interpolation
        ground_truth_hr = lin_interp(ground_truth_hr)
        ground_truth_spo2 = lin_interp(ground_truth_spo2)

        return ground_truth_hr, ground_truth_spo2,  ground_truth_time_stamps, None


#nearest neighbour interpolation if data isnt correctly read in beginning/end of stream
def nn_interp_tails(data, reverse=False):
    #if list is reversed, then end is nn interpolated
    if reverse: data.reverse()

    if data[0]==0:
        idx =  np.nonzero(data)[0][0]
        data_interpolated = [data[idx] for val in data[:idx]]
        data[:idx] = data_interpolated

    #reverse list back to normal order again
    if reverse: data.reverse()

    return data

#linear interpolation if data isnt correclty read in the middle of stream
def lin_interp (data):
    #make 0 values to None
    data = [None if val==0 else val for val in data]
    #interpolate none values
    data = pandas.DataFrame(data).interpolate()

    return data[0].tolist()


def get_fps_from_video_time_stamps_file (path):
    data = pandas.read_csv(path+"/video_time_stamps.csv", "r", delimiter=",")
    time_stamps = data["time_stamps"].tolist()

    fps = len(time_stamps)/time_stamps[-1]
    return fps

def live_plot_pulse_signal (pulse_signal):
    if not plt.get_fignums():
        plt.plot(pulse_signal)
        plt.ylabel("Amplitude")
        plt.xlabel("Frames")
        plt.title("rPPG Signal")
        plt.show()
    else:
        plt.clf()
        plt.plot(pulse_signal)
        plt.draw()

class GPUBuffer():
    def __init__(self, buffer_size, device="cpu"):
        self.device = device
        self.buffer = torch.tensor([], device=self.device)
        self.size = buffer_size
        self.buffer = []

    def append(self,data):
        data = torch.tensor(data, device=self.device)
        self.buffer.append(data)


    def get_numpy(self):
        return [entry.to(torch.device("cpu")).numpy() for entry in self.buffer]

    def pop(self, idx):
        self.buffer.pop(idx)

    def is_full (self):
        if len(self.buffer)>=self.size:
            return True
        else:
            return False


def save_rgb_signals (signal, hr, subject, path = "/media/BAB1-9078/raw_rgb_signals/"):
    fields = ["r", "g", "b", "hr"]
    rows = [ (rgb[0][0], rgb[0][1], rgb[0][2] , hr_val) for rgb, hr_val in zip(signal, hr)]
    with open(path+"subject"+str(subject)+'.csv', 'a') as file:
        write = csv.writer(file)
        write.writerow(fields)
        write.writerows(rows)
