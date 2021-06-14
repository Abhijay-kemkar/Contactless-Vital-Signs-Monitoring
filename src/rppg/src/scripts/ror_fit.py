import pdb
import numpy
import scipy
import csv
import numpy as np
import matplotlib.pyplot as plt


filename = "spo2_calibration_unfiltered.csv"
results = np.zeros((1,7))
with open ('rppg/calibration/' + filename, 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        r = np.array(list(map(float, line)))
        results = np.vstack( (results, r ) )


#drop zero entry
results = results[1:]

spo2_gt, spo2_pred = results[:,4], results[:,5]
ror = results[:,6]
lin_parameter = np.polyfit(ror, spo2_gt, 1)
b,a = lin_parameter[0], lin_parameter[1]
print(b,a)