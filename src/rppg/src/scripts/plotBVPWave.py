import os
import matplotlib.pyplot as plt
from utils import *
import argparse
import pdb
import cv2
import pickle
from rppg.pbv import *

data_path ='/media/7F9C-1517/dataset/'
parser = argparse.ArgumentParser(description='Process some inputs.')

parser.add_argument('-rppg', default =True, action ='store_false')
parser.add_argument('-raw_roi', default =False, action ='store_true')
parser.add_argument('-ubfc', default =False, action ='store_true')
parser.add_argument('--patch', default =500)
parser.add_argument('--fs', default = 30)

args = parser.parse_args()

rppg = args.rppg
getRawRoi = args.raw_roi
ubfc = args.ubfc
fs =  args.fs

dir_subjects = []
for (dirpath, dirnames, filenames) in os.walk(data_path):
    # checks if filenames contains a valid video extension
    if not dirpath == data_path:
        dir_subjects.append(dirpath+"/")

    #dir_subjects = [x+"/" for x in dir_subjects[0]]
subjects = []
pbv = PbvEstimator()

for dir_subject in dir_subjects:
    subject_nr = dir_subject.split("subject")[-1].replace('/', '')
    subjects.append(subject_nr)

    path = dir_subject

    # read gt and convert string to flaots
    ground_truth_file = open(path + 'ground_truth.txt', "r")
    ground_truth_str = ground_truth_file.read().split()
    ground_truth = [float(i) for i in ground_truth_str]

    #cap = cv2.VideoCapture(dir_subject+"vid.avi")
    #video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # split ground truth in signal and hr
    gt_length = len(ground_truth) // 3
    if not rppg:
        signal = ground_truth[:gt_length]
    else:
        if ubfc:
            if getRawRoi:
                path = 'results/ubfc_results/roi_'
                file = path + str(subject_nr)+'.p'
                with open(file, 'rb') as file:
                    roi = pickle.load(file)['roi']
                    roi_length = len(roi)

                roi = pbv._sliceIntoSubROIs(roi, n_roi_slices=3)
                signal = pbv.getMeanRGBSignal(roi)
            else:
                file  = 'prediction.p'
                with open(path+file, 'rb') as file:
                    prediction = pickle.load(file)
                    roi_length = len(prediction)
                signal, _, time_stamps = prediction['signal_avg'], prediction['fps'], prediction['time_stamps']
                signal = np.stack(signal).squeeze()

        else:
            if getRawRoi:
                with open('results/roi.p', 'rb') as file:
                    roi = pickle.load(file)['roi']
                    roi_length = len(roi)
                signal = pbv.getMeanRGBSignal(roi, multi_roi_slices=True, roi_slices=1)


        hr, sp02, signal, fft, freq = pbv.estimateHR_Sp02(signal, fs = fs, filter =True)
        signal  = np.mean(signal[:,:,0], axis=1)
        fft = np.mean(fft[:,:,0], axis=1)

        #hr, freq, fft, signal, freq = pbv.estimateHR(signal, fs)

        ground_truth_hr = ground_truth[gt_length:2 * gt_length][:roi_length]
        most_common_hr = max(set(ground_truth_hr), key=ground_truth_hr.count)
        #pdb.set_trace()
        most_common_hr_idx = (np.abs(fft-most_common_hr/60)).argmin()
        print('Subject {:} Estimated HR: {:}, Target HR: {:}'.format(subject_nr, hr, most_common_hr))

        f, (ax1, ax2) = plt.subplots(2, 1)
        f.suptitle('Subject {:}, most common HR {:}'.format(subject_nr, most_common_hr))
        ax1.plot(signal)
        ax2.plot(freq, fft)
        #ax2.vlines(most_common_hr_idx, 0, 1, linestyles='dashed', colors='r')
        #ax1.plot(signal)
        #ax2.plot(freq, fft_rppg)
        ax1.set_title('Time Domain')
        ax2.set_title('Frequency Domain, Est HR {:}'.format(hr))
        ax1.set(xlabel='time [s]', ylabel='amplitude')
        ax2.set(xlabel='frequency [hz]', ylabel='amplitude')
        plt.tight_layout()
        if rppg:
            fn = 'rppg_'
        else:
            fn = 'ppg_'

        plt.savefig('./results/ubfc_results/' + fn +
                    subject_nr + '.png')

