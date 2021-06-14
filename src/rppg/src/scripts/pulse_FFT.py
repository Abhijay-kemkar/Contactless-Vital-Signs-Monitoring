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

        spectrum, freq, max_hr_hz = rppg.getFFTSpectrumMax(pulse_signal, fps_video)



        plt.plot(freq , spectrum ,color = 'red')

        plt.show()

        sample_start += slide
        sample_end += slide
