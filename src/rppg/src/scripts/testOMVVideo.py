import jetson.utils

display = jetson.utils.glDisplay()
screen_resolution = (display.GetWidth(), display.GetHeight())
from utils import *
from FaceDetection import Detector
from imutils import face_utils
from faceTracking.rppg.rppg import *

network_backbone="mobile"
device = torch.device("cuda")

face_detector = Detector(network_backbone=network_backbone, face_detection_resolution=FACE_DETECTION_RESOLUTION)
face_detector.loadToDevice(device, toTensorRT=False)
# get landmark detector
landmark_detector = dlib.shape_predictor(TRAINED_MODEL_LANDMARKS)

file = "/home/philipp/Masterthesis/faceTracking/results/distance_img/openmv_240p.mp4"

reader = cv2.VideoCapture(file)
status, frame = reader.read()
buffer_size = 600
fs = 20
sliding_window_stride = 150
roi_buffer = []
sliding_window_count = 0
hr_results, spo2_results = [], []
i=0
rppg = RppgEstimator()

while status:

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bbox = face_detector.forward(frame)
    frame_reshaped = cv2.resize(frame, screen_resolution)
    #img_cuda = jetson.utils.cudaFromNumpy(frame_reshaped)
    #display.RenderOnce(img_cuda)

    print("i", i)
    i+=1
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

            if roi.shape[0] > 1 and roi.shape[1] > 1 \
                    and len(roi_buffer) < buffer_size+1:
                roi_buffer.append(roi)

            if len(roi_buffer) == buffer_size:
                roi_buffer.pop(0)

    sliding_window_count += 1
    if len(roi_buffer) >= buffer_size-1 and sliding_window_count >= sliding_window_stride:
        print('Processing rppg ...')
        mean_rgb, mean_rgb_sliced = rppg.getMeanRGBSignals(roi_buffer, n_roi_slices=1, n_channels=3)
        vital_signs = rppg.estimateVitalSignsInThread(mean_rgb, 0, fs=fs)
        hr = vital_signs['hr']
        spo2 = vital_signs['spo2']
        hr_results.append(hr)
        spo2_results.append(spo2)
        sliding_window_count = 0
        print(hr)

    status, frame = reader.read()



print("window length", buffer_size/fs)
print("stride length", sliding_window_stride/fs)
print("hr results:", hr_results)
print("Ground truth HR was measured ~65 bpms")

pdb.set_trace()