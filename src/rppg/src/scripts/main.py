from __future__ import print_function
import argparse
import traceback
import jetson.utils
import queue

from imutils import face_utils

from inputstream.flir.blackfly import Blackfly

# for some reason, initing the display fails if it is imported after FaceDetection, utils and torch
display = jetson.utils.glDisplay()
from FaceDetection import Detector
from utils import *
import torch
import pickle
from faceTracking.rppg.rppg import *

# get screen resolution
#TODO: find work around when there are two screens
screen_resolution = (display.GetWidth(), display.GetHeight())

# argumetns for inptu
parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('-save', action='store_true', default=False, help='save video or not')
parser.add_argument('-show', action='store_true', default=False, help='show video or not')
parser.add_argument('--detection_rate', default=10, type=int, help='use only every n-th frame for face detection')
parser.add_argument('--resolution', default="openmv", help='change general resolution')
parser.add_argument('--fd_resolution', default=FACE_DETECTION_RESOLUTION,
                    help='resoultion for images for face detection')
parser.add_argument('--cam', default='blackfly',
                    help='which camera to use: choose from ["blackfly"]. ("openmv" and "video" yet to be implemented)')
parser.add_argument('--length', default=float('inf'), help='maximum length of stream')
parser.add_argument('--skin_segm', default='none',
                    help='skin segmenation method in ROI. Choose form [hsv, conaire, none]')
parser.add_argument('-static_face_bb', action='store_true', default=False,
                    help='determines if face bounding box rotates with roll angle of face')
parser.add_argument('-blacken', action='store_true', default=False,
                    help='determines if area outside face bounding box is blackened or not')
parser.add_argument('-useRT', action='store_true', default=False, help='enables tensorRT porting for face detector')
parser.add_argument('--network', default='mobile',
                    help='choose from face detection network backbone [resnet, mobile].Default is mobile')
parser.add_argument('--roi_slices', default=10, help='the roi is split into n^2 equal subregions.  ')
parser.add_argument('--buffer_size', default=300, help='Buffer size for rppg rignal estimation')
args = parser.parse_args()

useCPU = False

if __name__ == '__main__':
    # Retrieve singleton reference to system object

    # get input parameters
    save_video = args.save
    detection_rate = args.detection_rate
    show_video = args.show
    static_face_bb = args.static_face_bb
    skin_segmentation_method = args.skin_segm
    blackenImageOutsideFace = args.blacken
    useTensorRT = args.useRT
    face_detection_resolution = resolutionToShape(args.fd_resolution)
    network_backbone = args.network
    buffer_size = args.buffer_size
    roi_slices = int(args.roi_slices)
    
    if isinstance(args.resolution, str):
        resolution = resolutionToShape(args.resolution)
    elif isinstance(args.resolution, tuple):
        resolution = args.resolution
    maxStreamTime = args.length
    # if video not shown and maxstreamtime not defined: set maxstream time to 0 to ensure termination after 10 seconds
    if not show_video and args.length == float('inf'): maxStreamTime = 10

    # max resolution of screen
    pixels_screen = (3840, 2160)

    device = torch.device("cpu" if useCPU else "cuda")

    # get model and load to device
    face_detector = Detector(network_backbone=network_backbone, face_detection_resolution=face_detection_resolution)
    face_detector.loadToDevice(device, toTensorRT=useTensorRT)
    # get landmark detector
    landmark_detector = dlib.shape_predictor(TRAINED_MODEL_LANDMARKS)

    # init camera
    cam = Blackfly()

    # begin acquisiton
    cam.beginAcquistion(0)

    bbox = [[0, 0, 0, 0, 0]]
    _t = {'img_acquisition': Timer(), 'img_display': Timer(), 'landmark_det': Timer(), 'face_det': Timer(),
          'pose_estimation': Timer(), 'skin_segmentation': Timer(), 'rppg': Timer(), 'all': Timer()}

    # start loop
    frame_array = []
    pixel_areas = []
    count = 0
    countSecond_start = time.time()
    countSecond_diff = 0
    rppg = RppgEstimator()
    roi_buffer = queue.Queue(buffer_size)
    rppg_signals_avg = []
    rppg_signals_pulse = []
    time_stamps = []
    frames_counter = 0
    roi_buffer = []
    roi_buffer_sliced = []
    try:
        # loop until display will be closed or maxStreamTime is reached
        while display.IsOpen() and countSecond_diff <= float(maxStreamTime):
            frames_counter += 1
            _t['all'].tic()
            # get current frame
            _t['img_acquisition'].tic()
            frame = cam.getFrame()
            _t['img_acquisition'].toc()
            time_stamps.append(countSecond_diff)

            # change general resoultion
            frame = cv2.resize(frame, resolution)

            # forward pass through face detection networks
            if count == 0 or count == detection_rate:
                _t['face_det'].tic()
                bbox = face_detector.forward(frame)
                _t['face_det'].toc()

                count = 0  # count gets incremented at the end

            # detect & draw landmarks (+face box) for each detected face
            if len(bbox) > 0:  # if statement only to correctly measure time
                for b in bbox:
                    if b[4] < VIS_THRESH:
                        continue
                    # detect landmarks
                    _t['landmark_det'].tic()
                    rectangle = dlib.rectangle(left=b[0], top=b[1], right=b[2], bottom=b[3])
                    shape_lm = landmark_detector(frame, rectangle)
                    _t['landmark_det'].toc()

                    # transform to numpy array
                    shape = face_utils.shape_to_np(shape_lm)

                    # draw landmarks
                    for (x, y) in shape: cv2.circle(frame, (x, y), 1, (255, 0, 255), 3)

                    # estimate angles of face
                    _t['pose_estimation'].tic()
                    poseEstimator = PoseEstimator(shape_lm, frame.shape, cam.focal_length)
                    pitch_angle, yaw_angle, roll_angle = poseEstimator.getEulerAngles()
                    _t['pose_estimation'].toc()

                    # quantize roll angle
                    roll_angle = int(roll_angle)

                    # get forehead ROI points
                    roi = getROICoordinates(b, shape)
                    forehead = cropImage(frame, roi)

                    # store the forehead crops in a fifo stack
                    # TODO: calculate rppg signal with fifo buffer in realtime
                    if forehead.shape[0] > roi_slices and forehead.shape[1] > roi_slices:
                        # store the forehead crops in a fifo stack
                        _t['rppg'].tic()
                        #forehead = pbv.preprocessROI(forehead, n_roi_slices=roi_slices)
                        #rppg_signal_avg = pbv.getNormalizedSignal(forehead)
                        #rppg_signals_avg.append(rppg_signal_avg)
                        _t['rppg'].toc()

                        if len(roi_buffer) < buffer_size:
                            roi_buffer.append(forehead)

                    b = list(map(int, b))
                    # draw rotated face bounding box
                    if abs(roll_angle) <= MAX_ROLL_ANGLE and not static_face_bb:
                        colorRectangle = (0, 255, 0)
                        box = rotateBbox(rectangle, roll_angle)
                        cv2.drawContours(frame, [box], 0, colorRectangle, 2)

                        # TODO: rotate and draw forehead ROI
                        # Problem: coordinates of rotated bbox must be adapted to to max y coordinate of forehead roi
                        # rectangle_roi  = dlib.rectangle(left=roi[0], top=roi[1], right=roi[2], bottom=roi[3])
                        # box_roi = rotateBbox(rectangle_roi, roll_angle)
                        # cv2.drawContours(frame, [box_roi], 0, (255, 255, 0), 3)
                        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 3)


                    # draw static face bounidng box
                    else:
                        colorRectangle = (255, 0, 0)
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), colorRectangle, 2)

                        # draw forehead ROI
                        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 3)

                    # draw text
                    text_confidence = "Conf.: {:.2f}".format(b[4])
                    pixel_area = (b[3] - b[1]) * (b[2] - b[0])
                    text_pixelArea = 'Area: ' + str(pixel_area)
                    text_angle = 'Angles: pitch {:.2f}, yaw {:.2f}, roll {:.2f}'.format(pitch_angle, pitch_angle,
                                                                                        roll_angle)
                    cx = b[0]
                    cy = b[1] + 12
                    cv2.putText(frame, text_confidence, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    cv2.putText(frame, text_pixelArea, (cx, cy + 18),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    cv2.putText(frame, text_angle, (cx, cy + 36),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # list of pixel areas
                    pixel_areas.append(pixel_area)

                # skin segmentation
                _t['skin_segmentation'].tic()
                frame, _ = skinSegmentation(frame, bbox, blacken_img=blackenImageOutsideFace,
                                            skin_segmentation_method=skin_segmentation_method)
                _t['skin_segmentation'].toc()
                # only show skin from skin segmentation algorithm

            # show fps from last frame in current frame. (img_display is shown from the last frame)
            for i, key in enumerate(_t):
                if _t[key].diff == 0:
                    fps = 0
                else:
                    fps = 1 / _t[key].diff
                fpsString = '{:} :  {:10.2f}'.format(key, fps)
                y_coord = 50 + i * 18
                cv2.putText(frame, fpsString, (50, y_coord), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # for saving video later
            if save_video: frame_array.append(frame)

            # downsize frame s.t. it fits on screen
            _t['img_display'].tic()
            if show_video:
                frame = cv2.resize(frame, pixels_screen)
                # render image in display
                img_cuda = jetson.utils.cudaFromNumpy(frame)
                display.RenderOnce(img_cuda, MODE_SETWIDTH, MODE_SETHEIGHT)
            _t['img_display'].toc()
            # print('Display FPS: ', display.GetFPS())

            time_overall = _t['all'].toc()
            print('Overall FPS: {:.2f}'.format(1 / time_overall['diff']))

            # update time difference
            countSecond_diff = time.time() - countSecond_start

            count += 1

        # save video
        if save_video:
            fps = int(1 / _t['all'].avg)
            size = resolution
            out = cv2.VideoWriter('results/video.avi', cv2.VideoWriter_fourcc(*'MP42'), fps, size)
            # out = WriteGear(output_filename='results/video.mov', logging=True)
            for i in range(len(frame_array)):
                print(i + 1, '/', len(frame_array))
                # writing to a image array
                out.write(frame_array[i])
            # out.close()
            out.release()

        #calculate hr and sp02
        """roi_buffer_processed = pbv.preprocessROI(roi_buffer, n_roi_slices=roi_slices)
        signal = pbv.getNormalizedSignal(roi_buffer_processed)
        hr, sp02, _, _, _ = pbv.estimateHR_Sp02(signal, filter=False)"""

        #save rppg signal and recorded roi
        rppg_signal_save = {'fps': 1 / _t['all'].getAvg(), 'signal_avg': rppg_signals_avg,
                            'time_stamps': time_stamps}
        pickle.dump(rppg_signal_save, open('results/rppg_signal.p', "wb"))

        roi_buffer_save = {'roi': roi_buffer}
        pickle.dump(roi_buffer_save, open('results/roi.p', "wb"))

        # compile all FPS into a string for display in next frame
        avgFPS = '\nImg Acquisiton.: {:2f} \nFace Detection.: {:2f} \nLandmark Detection.: {:2f} \nPose Estimation: {:2f} \n' \
                 'Skin Segmentation: {:2f}\nrPPG: {:2f}\nImg Display.: {:2f}\nOverall: {:2f}\n '.format(
            1 / _t['img_acquisition'].getAvg(),
            1 / _t['face_det'].getAvg(),
            1 / _t['landmark_det'].getAvg(),
            1 / _t['pose_estimation'].getAvg(),
            1 / _t['skin_segmentation'].getAvg(),
            1 / _t['rppg'].getAvg(),
            1 / _t['img_display'].getAvg(),
            1 / _t['all'].getAvg())

        print(avgFPS)
        print('Detection Rate: ', detection_rate)
        print('Original Resolution: ', resolution)
        print('Avg. Face Area: ', np.mean(pixel_areas))
        print('Time [s]: ', int(countSecond_diff))
        print('number of frames:', frames_counter)
        #print('Estimated HR: ', hr)
        #print('Esitamted Sp02: ', sp02 )

        # end acquistion and deinit cam
        cam.endAll()
        # del instance of cam
        del cam

    except Exception as e:
        # end and release instances
        cam.endAll()
        del cam

        print(traceback.format_exc())
