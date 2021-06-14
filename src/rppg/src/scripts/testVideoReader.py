import pdb
import cv2
from config import *
import jetson.utils
display = jetson.utils.glDisplay()
screen_resolution = (display.GetWidth(), display.GetHeight())
print("screen resolution", screen_resolution)
from FaceDetection import Detector
from utils import *
from imutils import face_utils
import random

network_backbone="mobile"
device = torch.device("cuda")
face_detector = Detector(network_backbone=network_backbone, face_detection_resolution=FACE_DETECTION_RESOLUTION)
face_detector.loadToDevice(device, toTensorRT=False)
# get landmark detector
landmark_detector = dlib.shape_predictor(TRAINED_MODEL_LANDMARKS)

#data_path="/media/BAB1-9078/vid.avi"
#data_path = "/media/philipp/Samsung_T5/test/{:}.avi".format("RGBA")
dir_subjects = getDirToVideos(BWH_SMALL_PATH, GOOD_SUBJECTS_BWH_SMALL)
dir_subjects = random.sample(dir_subjects, len(dir_subjects))
start =  time.time()
for dir_subject in dir_subjects:
    reader = cv2.VideoCapture(dir_subject+"/vid.avi")
    status, frame = reader.read()
    i,j = 0,0
    while status:
        if i ==0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (screen_resolution))
            bbox = face_detector.forward(frame)
            i=10
        else:
            i-=1

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
                #draw roi coord
                cv2.rectangle(frame, (roi_coord[0], roi_coord[1]), (roi_coord[2], roi_coord[3]), (255, 255, 0), 3)

        #resize to screen
        try:
            frame_reshaped = cv2.resize(frame, screen_resolution)
            img_cuda = jetson.utils.cudaFromNumpy(frame)
            display.RenderOnce(img_cuda)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t1 = time.time()
            status, frame = reader.read()
            print("img acq.", 1 / (time.time() - t1))



        except Exception as e:
            print(e)
