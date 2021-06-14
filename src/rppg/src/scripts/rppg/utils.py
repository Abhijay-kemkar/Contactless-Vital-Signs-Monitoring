import pdb

import numpy as np
from scipy.signal import find_peaks
from rppg.config import *
import os

def peak_to_through(signal, distance):
    max_peaks = find_peaks(signal, distance=distance)[0]
    min_peaks = find_peaks(-signal, distance=distance)[0]

    #drop first entry of min_peaks if first peak is a min peak
    if min_peaks[0]<max_peaks[0]: min_peaks = min_peaks[1:]

    #make them equally lengthed
    length = min(len(max_peaks), len(min_peaks))
    max_peaks = max_peaks[:length]
    min_peaks = min_peaks[:length]

    #calcuate peak to through height diff
    pth_heights =signal [max_peaks] - signal[min_peaks]

    return np.average(pth_heights)

def get_ror(spat_mean_rgb, ror_parameter=ROR_PARAMETERS, fs=30,
                 moving_avg_window_size=ROR_DC_MVG_AVG_WINDOWS_SIZE):
    red_signal = spat_mean_rgb[:, 0]
    blue_signal = spat_mean_rgb[:, 2]

    # moving average = dc component
    dc_red = np.ma.average(red_signal)
    dc_blue = np.ma.average(blue_signal)

    min_peak_distance = fs / MAX_HEART_HZ

    # get ac componenmts
    ac_red = peak_to_through(red_signal, min_peak_distance)
    ac_blue = peak_to_through(blue_signal, min_peak_distance)

    a, b = ror_parameter[0], ror_parameter[1]
    ror = (ac_red / dc_red) / (ac_blue / dc_blue)

    return ror