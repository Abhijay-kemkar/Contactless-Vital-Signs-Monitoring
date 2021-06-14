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

ext= 'pos_wholeface'
ext = 'pos_forehead'
ext = 'pos_crop'
#ext= 'pos_wholeface'
ext = "pos_low_res" #320x280 pixel
ext = "pos_low_res_2" #180x135 pixel
ext = "pos_1080p" #1920x1080 pxiels
ext = "pos_ubfc_res"
ext = "pos_ubfc_10s"
ext = "pos_ubfc_15s"
ext = "pos_ubfc_5s"
ext = "pos_ubfc_30s"
ext = "pos_ubfc_res"
ext = "pos_skinsegm"

mode = '_all'

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

#drop zero entry
results = results[1:]

#extract ground truth and predictions
gt_hr_all =  np.reshape(results[:, [2]], results[:, [2]].shape[0])
pred_hr_all =  np.reshape(results[:, [3]], results[:, [3]].shape[0])

errors = []
first_line_of_subjects = True
previous_subject = -1
per_subject_err = []
per_subject_mae = []
per_subject_rmse = []
subjects = []
for measurement in results:
    subject = measurement[0]
    err = measurement[3] - measurement[2]
    if first_line_of_subjects or previous_subject == subject:
        errors.append(err)
        first_line_of_subjects= False
    else:
        per_subject_err.append(np.average(errors))
        mae = np.sum(np.abs(errors)) / len(errors)
        rmse =  math.sqrt(np.sum(np.square(errors)) / len(errors))
        per_subject_mae.append(mae)
        per_subject_rmse.append(rmse)
        subjects.append(previous_subject)
        errors = []
        errors.append(err)
        first_line_of_subjects = True

    previous_subject= subject

#calculate mae and rmse for all subjects
overall_mae = np.sum(np.abs(per_subject_err)) / len(per_subject_err)
overall_rmse = math.sqrt(np.sum(np.square(per_subject_err)) / len(per_subject_err))

#overall_mae = np.average(per_subject_mae)
#overall_rmse = np.average(per_subject_rmse)

#order according to subjects number
per_subject_mae = [x for _, x in sorted(zip(subjects, per_subject_mae))]
per_subject_rmse = [x for _, x in sorted(zip(subjects, per_subject_rmse))]


# plot results
width = 0.3
subjects = np.array(subjects)

x_axis = np.arange( 1,len(subjects)+1 )
plt.bar(x_axis-0.15, per_subject_mae, width=width, color="r")
plt.bar(x_axis+0.15, per_subject_rmse, width=width, color="b")
plt.title('Overall MAE: {:.1f}, Overall RMSE: {:.1f}'.format(overall_mae, overall_rmse))
plt.xticks(x_axis, np.sort(subjects).astype(int))
plt.xlabel('Subject ID')
plt.ylabel('HR [bpms]')
plt.legend()
plt.savefig('./results/ubfc_results/results_{:}.png'.format(ext))
plt.show()
plt.clf()




