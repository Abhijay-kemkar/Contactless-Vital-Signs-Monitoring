import matplotlib.pyplot as plt
import csv
from utils import *
import hrvanalysis
from scipy.signal import correlate
import numpy as np
from scipy.signal import argrelextrema
import cv2
print(cv2.getBuildInformation())
pdb.set_trace()

dataset = "ubfc"
subejct = 34
data_path = UBFC_PATH
dir_subject = data_path+"subject"+str(subejct)+"/"
ground_truth_hr, ground_truth_spo2, gt_ppg = load_ground_truth_values(dir_subject, dataset, 0)
"""
gt_ppg=np.array(gt_ppg)
#method was adapted from https://blog.orikami.nl/exploring-heart-rate-variability-using-python-483a7037c64d
# linear spaced vector between 0.5 pi and 1.5 pi
t = np.linspace(0.5 * np.pi, 1.5 * np.pi, 15)
# use sine to approximate QRS feature
qrs_filter = np.sin(t)
threshold=0.3

similarity = correlate(gt_ppg, qrs_filter) > threshold
peaks = gt_ppg[similarity[len(gt_ppg)]][0]
peaks= argrelextrema(peaks, np.greater)[0]

for x_coord in peaks:
    y_max = gt_ppg[x_coord]
    plt.axvline(x=x_coord, color='r', linestyle='-')
plt.plot(gt_ppg)"""
plt.plto(ground_truth_hr)

#plt.plot(ground_truth_hr)
plt.title("Subject {:} HR Ground Truth".format(subejct))
plt.xlabel("Time [frames]")
plt.ylabel("HR [bpms]")
plt.legend(["peaks", "ppg signal"])
plt.show()