from rppg.rppg import *
from numpy import genfromtxt
import numpy as np 
import glob
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import *
from math import *
import time 
import pdb

file_directory = "/home/naitik/Work/MIT/raw_rgb_signals/*"
file_extension = ".csv"

input_paths = glob.glob(file_directory+file_extension)

true_hr = [] 
predicted_hr = []

mae_samplewise = []
rmse_samplewise =  []

X_input_samples = []

time_FFT = []
time_p2p = []

for i in range(len(input_paths)):
    my_data = genfromtxt(input_paths[i], delimiter=',')
    
    subject_num = input_paths[i].replace("/home/naitik/Work/MIT/raw_rgb_signals/subject","")
    subject_num = subject_num.replace(".csv","")
    X_input_samples.append(int(subject_num))
    
    my_data = my_data[1:]
    X = np.zeros((my_data.shape[0],3))
    y = np.zeros((my_data.shape[0],1))
    
    # print(input_paths[i])
    for i in range(my_data.shape[0]):
        X[i][0] = my_data[i][0]
        X[i][1] = my_data[i][1]
        X[i][2] = my_data[i][2]
        y[i][0] = my_data[i][3]
    
    rppg = RppgEstimator()

    sample_start = 0
    sample_end = 600
    
    slide = 30

    
    # print(len(X))

    true_hr_sample = []
    predicted_hr_sample = []

    while sample_end < len(X) :
        # print(sample_start)
        # print(sample_end)
        mean_rgb = X[sample_start:sample_end]
        mean_rgb_sliced = None
        fps_video = 30
        pulse_signal_method = "pos"
        spo2_pulse_signal_method = None

        start_time_fft = time.time()

        result , pulse_signal = rppg.estimateVitalSigns(mean_rgb , mean_rgb_sliced, fs=fps_video, method=pulse_signal_method, spo2_method= spo2_pulse_signal_method)

        end_time_fft = time.time()
        
        #pdb.set_trace() 
        
        #start_time_p2p = time.time()

        pulse_signal = pulse_signal.flatten()
        # plt.plot(pulse_signal)
        
        #print(pulse_signal)
        peaks, _ = find_peaks(pulse_signal, height=0)

        # plt.plot(peaks, pulse_signal[peaks], "x")
        # plt.show()

        threshold = 0.000

        true_peaks = [] 
    
        for i in range(len(peaks)) :
            if pulse_signal[peaks[i]] > threshold :
                true_peaks.append(peaks[i])
        
        avg_time_interval = 0 

        for i in range(len(true_peaks)-1):
            frames = true_peaks[i+1] - true_peaks[i]
            ins_time_interval = frames/fps_video
            avg_time_interval += ins_time_interval
    
        avg_time_interval /= (len(true_peaks)-1)

        hr_p2p = (1/avg_time_interval)*60

        end_time_p2p = time.time()

        time_FFT.append(end_time_fft-start_time_fft)
        print(1/(end_time_fft-start_time_fft))

        time_p2p.append(end_time_p2p-start_time_fft)
        #print(1/(end_time_p2p-start_time_fft))

        hr_true = (sum(y[sample_start:sample_end])/150)[0]

        # truth_vals = y[sample_start:sample_end].flatten()
        # truth_vals = truth_vals.astype(int)
        # hr_true = np.bincount(truth_vals).argmax()

        # print(hr_p2p)
        # print(hr_true)

        true_hr.append(hr_true)
        predicted_hr.append(hr_p2p)

        true_hr_sample.append(hr_true)
        predicted_hr_sample.append(hr_p2p)

        sample_start += slide 
        sample_end += slide
    
    mae_sample = mean_absolute_error(true_hr_sample, predicted_hr_sample)
    rmse_sample = sqrt(mean_squared_error(true_hr_sample, predicted_hr_sample))

    mae_samplewise.append(mae_sample)
    rmse_samplewise.append(rmse_sample)


mae = mean_absolute_error(true_hr, predicted_hr)
rms = sqrt(mean_squared_error(true_hr, predicted_hr))

average_time_fft = sum(time_FFT) / len(time_FFT)
average_time_p2p = sum(time_p2p) / len(time_p2p)

X = X_input_samples
Y1 = mae_samplewise
Y2 = rmse_samplewise

zipped_lists = zip(X,Y1,Y2)

sorted_zipped_lists = sorted(zipped_lists)

X.clear()
Y1.clear()
Y2.clear()

for x, y in enumerate(sorted_zipped_lists) :
    X.append(y[0])
    Y1.append(y[1])
    Y2.append(y[2])
    
# X = X_input_samples

# zipped_lists = zip(X,Y2)

# sorted_zipped_lists = sorted(zipped_lists)

# X.clear()
# Y2.clear()

# for x, y2 in enumerate(sorted_zipped_lists) :
#     X.append(y2[0])
#     Y2.append(y2[1])
  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, Y1, 0.4, label = 'MAE')
plt.bar(X_axis + 0.2, Y2, 0.4, label = 'RMSE')
  
plt.xticks(X_axis, X)
plt.xlabel("Subjects")
plt.ylabel("Error")
plt.title("Subjectwise Errors for " + pulse_signal_method)
plt.legend()
plt.show()

print(mae)
print(rms)
print("The average time  for execution in p2p is" , average_time_p2p )
print("The average time  for execution in fft is" , average_time_fft)
print("The average fps for p2p" , 1/average_time_p2p)
print("The average fps for fft" , 1/average_time_fft)
# print(len(mae_samplewise))
# print(len(rmse_samplewise))


#testing 1 : 0.5 seconds window stride
#testing 2 : 5 seconds window stride
#most common heart rate in the error : compare it with the average.
#