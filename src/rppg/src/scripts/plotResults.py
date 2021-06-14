import matplotlib.pyplot as plt
import csv
import numpy as np
import pdb
from utils import *
from config import *
import math


dataset = "bwhs_"
dataset = "ubfc_"

ext='bp_rgb'

ext=dataset+ext

results = np.zeros((1,6))
with open ('results/results_{:}.csv'.format(ext), 'r') as file:
    reader = csv.reader(file)
    subject = 0
    det_count = -1
    for line in reader:
        if not int(line[0])==subject:
            subject = int(line[0])
            det_count = -1
        det_count += 1#

        r = np.array(list(map(float, line)))
        results = np.vstack( (results, r ) )


#drop zero entry
results = results[1:]

#get per subjects errors
err_hr = results[:,2]-results[:,3]
err_spo2 = results[:,4]-results[:,5]
subjects = np.unique(results[:,0]).astype(int)

# calculate mae and rmse for all subjects
overall_mae_hr = np.sum(np.abs(err_hr)) / len(err_hr)
overall_rmse_hr = math.sqrt(np.sum(np.square(err_hr)) / len(err_hr))
overall_mae_spo2 = np.sum(np.abs(err_spo2)) / len(err_spo2)
overall_rmse_spo2 = math.sqrt(np.sum(np.square(err_spo2)) / len(err_spo2))

#order according to subjects number
per_subject_mae_hr, per_subject_rmse_hr = [], []
per_subject_mae_spo2, per_subject_rmse_spo2 = [], []
for s in subjects:
    idx = np.where(results[:,0]==s)
    err_hr = results[idx,2]-results[idx,3]
    err_spo2 = results[idx,4]-results[idx,5]
    per_subject_mae_hr.append(np.sum(np.abs(err_hr)) / len(err_hr[0]))
    per_subject_rmse_hr.append(math.sqrt(np.sum(np.square(err_hr[0])) / len(err_hr[0])))
    per_subject_mae_spo2.append(np.sum(np.abs(err_spo2[0])) / len(err_spo2[0]))
    per_subject_rmse_spo2.append(math.sqrt(np.sum(np.square(err_spo2[0])) / len(err_spo2[0])))


"""per_subject_mae_hr = [x for _, x in sorted(zip(subjects, per_subject_mae_hr))]
per_subject_rmse_hr = [x for _, x in sorted(zip(subjects, per_subject_rmse_hr))]
per_subject_mae_spo2 = [x for _, x in sorted(zip(subjects, per_subject_mae_spo2))]
per_subject_rmse_spo2 = [x for _, x in sorted(zip(subjects, per_subject_rmse_spo2))]"""

# plot results
width = 0.3
fig, axs = plt.subplots(2)
subjects = np.array(subjects)
x_axis = np.arange( 1,len(subjects)+1 )

axs[0].bar(x_axis-0.15, per_subject_mae_hr, width=width, color="r")
axs[0].bar(x_axis+0.15, per_subject_rmse_hr, width=width, color="b")
axs[0].set_title('Overall MAE: {:.1f}, Overall RMSE: {:.1f}'.format(overall_mae_hr, overall_rmse_hr))
axs[0].set_xticks(x_axis)
axs[0].set_xticklabels(np.sort(subjects).astype(int))
axs[0].set_xlabel('Subject ID')
axs[0].set_ylabel('HR [bpms]')

axs[1].bar(x_axis-0.15, per_subject_mae_spo2, width=width, color="r")
axs[1].bar(x_axis+0.15, per_subject_rmse_spo2, width=width, color="b")
axs[1].set_title('Overall MAE: {:.1f}, Overall RMSE: {:.1f}'.format(overall_mae_spo2, overall_rmse_spo2))
axs[1].set_xticks(x_axis)
axs[1].set_xticklabels(np.sort(subjects).astype(int))
axs[1].set_xlabel('Subject ID')
axs[1].set_ylabel('SpO2 Absolute Deviation [%]')

plt.legend()
plt.tight_layout()
plt.savefig('./results/ubfc_results/results_{:}.png'.format(ext))
plt.show()
plt.clf()


