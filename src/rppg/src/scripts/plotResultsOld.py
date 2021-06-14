import matplotlib.pyplot as plt
import csv
import numpy as np
import pdb
from utils import *
from config import *
import math

def butter_bandpass(signal, lowcut, highcut, fs, order=5, btype='band'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if btype == "low":
        freq = low
    else:
        freq = [low, high]

    b, a = butter(order, freq, btype=btype)

    y = lfilter(b, a, signal)
    return y



ext = 'hrr_filter'
ext = 'hrrOff_filter'
#ext = 'hrrOff_filterOff'
#ext = 'hrr_filterOff_best' # best result, but subjects seleciton was different
ext = 'hrrOff_filter_2'
ext = 'hrrOff_filter_3'
ext = 'hrrOff_filter_4' #cn filtered with wide band
ext = 'hrrOff_filter_5' #cn filtered with narrow band
#max to median shape
ext= 'hrr_filterOff_median'
ext = 'hrr_filterOff_copy' # best result, but couldnt be reprocudec
#best result patch_pulse filtered not cn
ext = 'hrrOff_filterOff'
#try with face crops = 1 and filter cn
ext= 'hrrOff_filter_slice1'
#1 roi slice, 1 sp02 value to check
ext = 'hrrOff_filterOff_1roi_1sp02'
#4 roi slice, 1 sp02 value to check
ext  = 'hrrOff_filterOff_1sp02'
#filter off, extended freq look up (45, 250) bpms
ext=  'hrrOff_filterOff_ext_hb'
ext = 'hrrOff_filterOff'
ext = 'hrrOff_filterOff_slice1_easyFFT'
ext = 'pbv'
ext ='chrom' #normalization
ext ='chrom_2' # no normalization
ext = 'chrom_3' #standard timewindow
ext = 'pbv_2'
ext = 'pbv_3' #slidin window 10s
ext = 'pos_2'
ext = 'pos_3' #roi slices 4
ext= 'chrom_4' #rois slices4
ext= 'comp' # with chrom
ext= 'ext_bbox250' # with chrom
ext= 'ext_bbox500' # with chrom
ext= 'pos_4' #1 roi slice, 20 sliding window
ext = 'pos'# mode all, time window 10s
ext = 'chrom' #mode all, time window 10s
ext= 'pos_2' #20s time window all, BEST
ext = 'chrom_2' #s20s window all
# try chrom, pos, pbv with new color ordering
ext='chrom_rgb_extBbox'
ext = 'pbv_rgb_extBbox'
ext= 'pos_rgb_extBbox'
ext = 'apbv_rgb_extBbox'
#
ext='chrom_rgb'
ext = 'pbv_rgb'
ext= 'pos_rgb'
ext = 'apbv_rgb'
# new resolution, and text is not in face
ext = 'pos_lr'
ext = 'pbv_lr'
ext = 'apbv_lr'
ext = 'chrom_lr'
# pos with 1s stride
ext = 'pos_stride30'
ext = 'pos_2'
# whole face is in lr
ext = 'pos_wholeface'
ext= 'pos_new' #buffersize 600
#ext= 'pos_new_2' #buffersize 300
# mode all
ext= 'pos_new_3' #with temporal norm on, bad
ext = 'pos_lr_crop'
ext = 'pos2' #evaluated on all
ext = 'pos2_2' # new roi slice 1 method
ext = 'pos_adapted' #new time avg.
#TODO: do exact same parameters with whole Face and Forehead roi to compare with paper
ext = 'pos_lr_crop_2' #with the exact same parameters as in paer (20s window, 0.5s update, bp filter)
ext = 'pos_crop' # adapted pos method
ext = 'pos_crop_whole' # now really all (NO SUBJECTS EXCLUDED!)
ext = 'pos_crop_3' #crop face + stride 150
ext = 'pos_detrend'
ext = 'chrom_detrend'
ext = 'pos_crop_avgGT' #from now own only avgGT
#TODO: order them correctly (named false)
ext= 'pos_wholeface'
ext = 'pos_forehead'
ext = 'pos_crop'
#ext= 'pos_wholeface'
ext = "pos_low_res" #320x280 pixel
ext = "pos_low_res_2" #180x135 pixel
ext = "pos_1080p" #1920x1080 pxiels
ext = "pos_ubfc_res"
ext = "pos_720p"
ext = "chrom"

ext = "pos_skinsegm"
#ext = "conaire"
mode = '_test'
mode = '_all'
#mode = '_train'
ext = ext+mode

results = np.zeros((1,4))

with open ('results/results_{:}.csv'.format(ext), 'r') as file:
    reader = csv.reader(file)
    subject = 0
    det_count = -1
    for line in reader:
        if not int(line[0])==subject:
            subject = int(line[0])
            det_count = -1
        det_count += 1#

        #TODO: maybe exclude partial covered foreheads as well?
        if not subject in BAD_SUBJECTS_UBFC:
            r =  np.array( (subject, det_count, float(line[1]), float(line[2])))
            results = np.vstack( (results, r ) )

"""
with open ('results/results_{:}.csv'.format(ext), 'r') as file:
    reader = csv.reader(file)
    subject = 0
    det_count = -1
    for line in reader:
        if not int(line[0]) == subject:
            subject = int(line[0])

    if not subject in BAD_SUBJECTS:
        r = np.array((subject, det_count, float(line[1]), float(line[2])))
        results = np.vstack((results, r))"""


#drop zero entry
results = results[1:]

#extract ground truth and predictions
gt_hr_all =  np.reshape(results[:, [2]], results[:, [2]].shape[0])
pred_hr_all =  np.reshape(results[:, [3]], results[:, [3]].shape[0])

#calculate mae and rmse subject wise, then mean it
#overall_mae = calculateMae(pred_hr_all, gt_hr_all)
#overall_rmse = calculateRmse(pred_hr_all, gt_hr_all)
error_per_subject = []
first_subject = True
previous_subject = -1
per_subject_err = []
per_subject_corr = []
for measurement in results:
    subject = measurement[0]
    if first_subject or previous_subject == subject:
        error_per_subject.append(measurement[2]-measurement[3])
        first_subject= False
    else:
        per_subject_err.append(np.average(error_per_subject))
        error_per_subject = []

    previous_subject= subject

overall_mae = np.sum(np.abs(per_subject_err)) / len(per_subject_err)
overall_rmse = math.sqrt(np.sum(np.square(per_subject_err)) / len(per_subject_err))
#calculate rmse per frame for plotting
result_pred = []
result_gt = []
max_det_count = max(np.reshape(results[:,[1]], len(results)))

for det in range(0, int(max_det_count+1) ):
    ind = results[:,1] == det

    gt_hr = results[ind, 2]
    pred_hr = results[ind, 3]

    pred_mean = np.mean(pred_hr)
    gt_mean = np.mean(gt_hr)

    result_pred.append(pred_mean)
    result_gt.append(gt_mean)

result_pred = np.array(result_pred)
result_gt = np.array(result_gt)

result_diff = np.abs(gt_hr_all-pred_hr_all)
# err below 6 bpms
low_err = len(np.where(result_diff<=6)[0])
high_err = len(np.where(result_diff>6)[0])
err_rate_1 = low_err / (low_err + high_err)

# err below 12 bpms
low_err = len(np.where(result_diff<=12)[0])
high_err = len(np.where(result_diff>12)[0])
err_rate_2 = low_err / (low_err + high_err)

 # plot results
plt.plot(result_pred, label='Prediction', linestyle='-')
plt.plot(result_gt, label='Ground Truth', linestyle='--')
# plt.plot(ground_truth_hr,  label = 'gt all', linestyle = '-')
plt.title('Overall MAE: {:.1f}, Overall RMSE: {:.1f} \n Acc < 6bpms: {:.2f}, '
          'Acc < 12 bpms: {:.2f}'.format(overall_mae, overall_rmse, err_rate_1, err_rate_2))
plt.xlabel('Detection')
plt.ylabel('HR [bpms]')
plt.legend()
plt.savefig('./results/ubfc_results/results_{:}.png'.format(ext))
plt.show()
plt.clf()




