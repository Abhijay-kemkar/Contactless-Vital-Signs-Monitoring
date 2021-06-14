from __future__ import print_function
import jetson.utils
# for some reason, initing the display fails if it is imported after FaceDetection, utils and torch
import pdb
display = jetson.utils.glDisplay()

import argparse
from imutils import face_utils
from FaceDetection import Detector
from utils import *
from rppg.rppg import *
import csv
import copy

# get screen resolution
screen_resolution = (display.GetWidth(), display.GetHeight())

if __name__ == '__main__':
    # Retrieve singleton reference to system object

    # get input parameters
    save_video = args.save
    detection_rate = args.detection_rate
    show_video = args.show
    static_face_bb = args.static_face_bb
    skin_segmentation_method = args.skin_segm
    blacken_img_outside_face = args.blacken
    useTensorRT = args.useRT
    network_backbone = args.network
    buffer_size = int(args.buffer_size)
    roi_slices = int(args.roi_slices)
    sliding_window_stride = int(args.stride)
    dataset = args.dataset
    max_rate_of_change_hr = int(args.hr_reset)
    save_plots = args.save_plots
    mode =  args.mode
    filter_pulse_signal = args.filter_patch
    filter_cn = args.filter_cn
    pulse_signal_method = args.method
    whole_face = args.whole_face
    temp_norm = args.temp_norm
    crop_face = args.crop_face
    roi_area= args.roi
    show_rppg_input_video = args.rppg_show
    fps_video = int(args.fs)
    spo2_pulse_signal_method = args.spo2_method
    if isinstance(args.resolution, str):
        resolution = resolutionToShape(args.resolution)
    elif isinstance(args.resolution, tuple):
        resolution = args.resolution
    if args.fd_resolution[0]*args.fd_resolution[1] < resolution[0]*resolution[1]:
        face_detection_resolution = args.fd_resolution
    else:
        face_detection_resolution = resolution
    if dataset=="ubfc":
        data_path = UBFC_PATH
        results_path = "results/ubfc_results/"
        subjects = GOOD_SUBJECTS_UBFC
        trials = []
    elif dataset == "bwh_small":
        data_path = BWH_SMALL_PATH
        subjects = GOOD_SUBJECTS_BWH_SMALL
        results_path = "results/bwh_small/"
        trials = list(range(1,BWH_SMALL_TRIALS+1))
    # change roi area to upper face if skin segm is actiavted
    if skin_segmentation_method == "none":
        skin_segmentation_method = None


    maxStreamTime = args.length
    # if video not shown and maxstreamtime not defined: set maxstream time to 0 to ensure termination after 10 seconds
    if not show_video and args.length == float('inf'): maxStreamTime = 10

    #face detection might give some bad results if exectued on the cpu
    device = torch.device("cuda")

    # get model and load to device
    face_detector = Detector(network_backbone=network_backbone, face_detection_resolution=face_detection_resolution)
    face_detector.loadToDevice(device, toTensorRT=useTensorRT)
    # get landmark detector
    landmark_detector = dlib.shape_predictor(TRAINED_MODEL_LANDMARKS)

    #use trail 1 of bwh dataset  as training set
    if mode == "train" and not dataset == "ubfc":
        exclude_trial = [2,3,4,5]
    else:
        exclude_trial = []

    # get all file names
    dir_subjects = getDirToVideos(data_path, subjects, exclude_trial=exclude_trial)

    bbox = [[0, 0, 0, 0, 0]]
    _t = {'img_acquisition': Timer(), 'img_display': Timer(), 'landmark_det': Timer(), 'face_det': Timer(),
          'pose_estimation': Timer(), 'skin_segmentation': Timer(), 'rppg': Timer(), 'all': Timer()}

    # start loop
    save_video_buffer = []
    pixel_areas = {"x":[], "y":[]}
    count = 0
    countSecond_start = time.time()
    countSecond_diff = 0
    hr_gt_all, hr_pred_all = [], []
    spo2_gt_all, spo2_pred_all = [], []
    #iterarte over subjects
    for dir_subject in dir_subjects:
        subject_nr = dir_subject.split("subject")[-1].split('/')[0]
        if dataset == "bwh_small":
            trial = int(dir_subject[-2])
        else:
            trial = 0

        ground_truth_hr, ground_truth_spo2, ground_truth_time_stamps, _ \
            = load_ground_truth_values(dir_subject, dataset, trial)

        video_filename = 'vid.avi'
        video_file = dir_subject + video_filename
        cap = cv2.VideoCapture(video_file)
        rppg_signals_avg = []
        time_stamps_rppg = []
        time_stamps = []
        roi_buffer = GPUBuffer(buffer_size)
        skin_segm_mask_buffer = []
        sliding_window_count = 1
        sliding_window_position = 0
        hr_gt_per_subject, hr_pred_per_subject = [], []
        rppg = RppgEstimator()
        countSecond_diff = 0
        if ground_truth_time_stamps:
            fps_pulse_oximeter = len(ground_truth_time_stamps) / ground_truth_time_stamps[-1]
        else:
            fps_pulse_oximeter = fps_video
        #fps_video = get_fps_from_video_time_stamps_file (dir_subject)
        #iterate over video until video is finished
        while True:
            _t['all'].tic()

            # get current frame
            _t['img_acquisition'].tic()
            status, frame = cap.read()
            if not status:
                print('End of video')
                #if you want to save mean rgb signals from video streams, choose buffer size to be very high and uncomment this
                #mean_rgb, mean_rgb_sliced = rppg.getMeanRGBSignals(roi_buffer.get_numpy(), n_roi_slices=1, n_channels=3, fs=fps_video)
                #save_rgb_signals(mean_rgb, ground_truth_hr, subject_nr)
                break
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _t['img_acquisition'].toc()
            #print("img acq", 1/_t['img_acquisition'].diff)

            # change general resoultion if its not none
            if resolution:
                frame = cv2.resize(frame, resolution)

            # forward pass through face detection networks
            if count == 0 or count == detection_rate:
                _t['face_det'].tic()
                bbox = face_detector.forward(frame)
                _t['face_det'].toc()
                #print("face det", 1 / _t['face_det'].diff)

                count = 0  # count gets incremented at the end

            # detect & draw landmarks (+face box) for each detected face
            if len(bbox) > 0:  # if statement only to correctly measure time
                for b in bbox:
                    #transfomr each bbox to [left, top, right, bot]
                    #b[1], b[3] = b[3], b[1]
                    if b[4] < VIS_THRESH:
                        continue
                    # detect landmarks
                    _t['landmark_det'].tic()
                    rectangle = dlib.rectangle(left=b[0], top=b[1], right=b[2], bottom=b[3])
                    landmarks_shape = landmark_detector(frame, rectangle)
                    _t['landmark_det'].toc()

                    # transform to numpy array
                    landmarks = face_utils.shape_to_np(landmarks_shape)

                    # draw landmarks
                    if show_video:
                        if not whole_face :
                            for (x, y) in landmarks: cv2.circle(frame, (x, y), 1, (255, 0, 255), 3)

                    # estimate angles of face
                    _t['pose_estimation'].tic()
                    # poseEstimator = PoseEstimator(shape_lm, frame.shape, 0)
                    pitch_angle, yaw_angle, roll_angle = 0, 0, 0
                    _t['pose_estimation'].toc()

                    # quantize roll angle
                    roll_angle = int(roll_angle)

                    # get forehead ROI points
                    if not whole_face:
                        roi_coord = getROICoordinates(b, landmarks, roi_area=roi_area)
                        # roi[2] += 500
                    #get whole or face crop as roi
                    else:
                        roi_coord = copy.copy( b[:-1])
                        # only take 60% of face
                        if crop_face: roi_coord = scaleBoundingBox(roi_coord, 0.6)
                        roi_coord[0], roi_coord[2], = roi_coord[2], roi_coord[0]
                    roi = cropImage(frame, roi_coord)

                    # store the forehead crops in a fifo stack (list)
                    # if its size is sufficently large and the buffer is not full
                    if roi.shape[0] > roi_slices and roi.shape[1] > roi_slices \
                            and not roi_buffer.is_full():
                        roi_buffer.append(roi)
                        time_stamps.append(countSecond_diff)

                    b = list(map(int, b))
                    # draw rotated face bounding box
                    if abs(roll_angle) <= MAX_ROLL_ANGLE and not static_face_bb:
                        colorRectangle = (0, 255, 0)
                        box = rotateBbox(rectangle, roll_angle)
                        if show_video:
                            cv2.drawContours(frame, [box], 0, colorRectangle, 2)
                            # TODO: rotate and draw forehead ROI
                            # Problem: coordinates of rotated bbox must be adapted to to max y coordinate of forehead roi
                            # rectangle_roi  = dlib.rectangle(left=roi[0], top=roi[1], right=roi[2], bottom=roi[3])
                            # box_roi = rotateBbox(rectangle_roi, roll_angle)
                            # cv2.drawContours(frame, [box_roi], 0, (255, 255, 0), 3)
                            cv2.rectangle(frame, (roi_coord[0], roi_coord[1]), (roi_coord[2], roi_coord[3]), (255, 255, 0), 3)


                    # draw static face bounidng box
                    else:
                        colorRectangle = (255, 0, 0)
                        if show_video:
                            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), colorRectangle, 2)

                            # draw forehead ROI
                            cv2.rectangle(frame, (roi_coord[0], roi_coord[1]), (roi_coord[2], roi_coord[3]), (255, 255, 0), 3)

                    # draw text
                    text_confidence = "Conf.: {:.2f}".format(b[4])
                    #pixel_area = (b[3] - b[1]) * (b[2] - b[0])
                    #text_pixelArea = 'Area: ' + str(pixel_area)
                    text_angle = 'Angles: pitch {:.2f}, yaw {:.2f}, roll {:.2f}'.format(pitch_angle, pitch_angle,
                                                                                        roll_angle)
                    cx = b[0]
                    cy = b[3] + 12
                    if show_video:
                        cv2.putText(frame, text_confidence, (cx, cy),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                        #cv2.putText(frame, text_pixelArea, (cx, cy + 18),
                        #            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                        cv2.putText(frame, text_angle, (cx, cy + 18),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # list of pixel areas
                    pixel_areas["x"].append(roi.shape[0])
                    pixel_areas["y"].append(roi.shape[1])

                    # skin segmentation
                    _t['skin_segmentation'].tic()
                    if not skin_segmentation_method is None:
                        skin_segm_mask = getSkinSegmentationMask(roi, method=skin_segmentation_method)
                        #change last entry of roi buffer to vector of pixels
                        roi_buffer[-1] = roi[skin_segm_mask]
                    _t['skin_segmentation'].toc()
                # only show skin from skin segmentation algorithm


            # end if statement over faces

            # show fps from last frame in current frame. (img_display is shown from the last frame)
            for i, key in enumerate(_t):
                if _t[key].diff == 0:
                    fps = 0
                else:
                    fps = 1 / _t[key].diff
                fpsString = '{:} :  {:10.2f}'.format(key, fps)
                y_coord = 50 + i * 18
                #cv2.putText(frame, fpsString, (50, y_coord), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # downsize frame s.t. it fits on screen
            _t['img_display'].tic()
            if show_video or show_rppg_input_video:
                frame = cv2.resize(frame, screen_resolution)
                # render image in display
                img = frame
                #img = np.stack((skin_segm_mask*255,) * 3, axis=-1)
                img_cuda = jetson.utils.cudaFromNumpy(img)
                #img_cuda = jetson.utils.cudaFromNumpy(roi)
                display.RenderOnce(img_cuda, MODE_SETWIDTH, MODE_SETHEIGHT)
                _t['img_display'].toc()
                #print("img display", 1 / _t['img_display'].diff)

            time_overall = _t['all'].toc()
            print('Overall FPS: {:.2f}'.format(1 / time_overall['diff']))

            # update time difference
            countSecond_diff = time.time() - countSecond_start

            # calculate hr and  sp02 for if buffer is full and sliding window is reached
            if roi_buffer.is_full() and (sliding_window_stride + 1) / sliding_window_count == 1:
                print('Processing rppg ...')
                _t['rppg'].tic()
                mean_rgb, mean_rgb_sliced = rppg.getMeanRGBSignals(roi_buffer.get_numpy(), n_roi_slices=1, n_channels=3, fs=fps_video)
                vital_signs, pulse_signal = rppg.estimateVitalSigns(mean_rgb, mean_rgb_sliced, fs=fps_video,
                                                                            method=pulse_signal_method, spo2_method= spo2_pulse_signal_method)
                _t['rppg'].toc()

                hr = vital_signs['hr']
                spo2 = vital_signs['spo2']
                ror =  vital_signs["ror"]
                idx_hr_slice_1 = sliding_window_position * (sliding_window_count - 1)
                idx_hr_slice_2 = buffer_size + idx_hr_slice_1
                idx_spo2_slice_1 = int(idx_hr_slice_1 * fps_pulse_oximeter / fps_video)
                idx_spo2_slice_2 = int(idx_hr_slice_2 * fps_pulse_oximeter / fps_video)
                #make exception if end of slice is reached
                try:
                    ground_truth_hr_slice = ground_truth_hr[idx_hr_slice_1:idx_hr_slice_2]
                    ground_truth_spo2_slice = ground_truth_spo2[idx_spo2_slice_1:idx_spo2_slice_2]
                except:
                    ground_truth_hr_slice = ground_truth_hr[buffer_size:-1]
                    if ground_truth_spo2:
                        ground_truth_spo2_slice = ground_truth_spo2[buffer_size*fps_pulse_oximeter / fps_video:-1]

                #compile hr
                hr_gt = int(np.average(ground_truth_hr_slice))
                hr_gt_all.append(hr_gt)
                hr_pred_all.append(hr)

                #make exception if no spo2 values are available from ground truth data
                try:
                    spo2_gt = findMostCommon(ground_truth_spo2_slice)
                    spo2_gt_all.append(spo2_gt)
                    spo2_pred_all.append(spo2)
                except Exception:
                    spo2_gt = 0
                    pass

                #live plots pulse signal
                if show_video:
                    live_plot_pulse_signal(pulse_signal)

                #append to results csv
                new_row = [subject_nr, trial, hr_gt, hr, spo2_gt, spo2, ror]
                if not mode == "train": new_row.pop(-1)
                with open('results/results.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow(new_row)

                # increment sliding window
                sliding_window_position += 1

            # reset sliding_window_counter
            if (sliding_window_stride + 1) / sliding_window_count == 1:
                sliding_window_count = 0
            # delete first entry of buffer
            if roi_buffer.is_full():
                roi_buffer.pop(0)

            # increment counter
            count += 1
            sliding_window_count += 1
            #end of video

        cap.release()

    # end of loop over subjects

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
    print('Avg. Face Area: ', np.mean(pixel_areas["x"]), "x", np.mean(pixel_areas["y"]))
    print('Time [s]: ', int(countSecond_diff))
    print('Original Resolution: ', frame.shape)

