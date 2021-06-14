import PySpin
import time
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.fftpack import fft
import argparse
import pickle
from scipy.signal import butter, lfilter
import numpy as np
import pdb
from utils import  *

parser = argparse.ArgumentParser(description='Process some inputs.')
parser.add_argument('-no_filter', action ='store_true',default = False)
parser.add_argument('--rppg', default = 'results/rppg_signal.p')
args = parser.parse_args()




if isinstance(args.rppg, str):
    rppg  =   pickle.load(open(args.rppg, "rb") )
else:
    rppg = args.rppg
#
low = 50 / 60
high = 150 / 60
fs = rppg['fps']


rppg_signal = np.array(rppg['signal'])
stream_length = rppg['time_stamps'][-1]

#resample signal
#n = int(fs*stream_length)
#rppg_signal = signal.resample(rppg_signal, n)

time_stamps = np.linspace(0, stream_length, num=len(rppg_signal))


outlier_indices =np.argwhere(rppg_signal>1)

if len(outlier_indices)>0:
    #linearly interpolate outliers

    for ind in outlier_indices:
        if ind == 0 :
            rppg_signal[0] = rppg_signal[1]

        elif ind == len(rppg_signal)-1:
            rppg_signal[-1] = rppg_signal[-2]

        else:
            rppg_signal[ind] = (rppg_signal[ind+1] - rppg_signal[ind-1])/2

if not args.no_filter:
    rppg_signal_low_passed = butter_bandpass(rppg_signal, low, high, fs)
    rppg_signal_low_passed_dc_normalized = np.divide(rppg_signal, rppg_signal_low_passed)
    rppg_signal = butter_bandpass(rppg_signal, low, high, fs)




hr, freq, fft_rppg = estimateHR (rppg_signal, fs, returnFFT=True)
print('Estimated Heart Rate: ', hr)

print()

f, (ax1, ax2) = plt.subplots(2,1)
f.suptitle('rPPG Signal. Estimated HR: {:}'.format(hr))
ax1.plot( rppg_signal)
ax2.plot(freq, fft_rppg)
ax1.set_title('Time Domain')
ax2.set_title ('Frequency Domain')
ax1.set(xlabel='time [s]', ylabel='amplitude')
ax2.set(xlabel='frequency [hz]', ylabel='amplitude')


plt.show()
