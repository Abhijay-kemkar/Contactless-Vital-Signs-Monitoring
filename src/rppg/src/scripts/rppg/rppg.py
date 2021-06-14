import pdb

import cv2
from scipy.signal import butter, lfilter
import math
from scipy.fftpack import fft, fftfreq
#from utils import *
from rppg.utils import *
from rppg.config import *
import threading

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

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


def cdf_filter(C_rgb, LPF, HPF, fs, bpf=False):
    """
    Color-distortion filtering for remote photoplethysmography.
    """
    L = C_rgb.shape[0]
    # temporal normalization
    Cn = C_rgb/np.average(C_rgb, axis=0) -1
    # Hanning Window
    # window = np.hanning(L).reshape(-1,1)
    # Cn = Cn * window
    #  FFT transform
    FF = fft(Cn, n=L, axis=0)
    freq = fftfreq(n=L, d=1/fs)
    # Characteristic transformation
    H = np.dot(FF, (np.array([[-1, 2, -1]])/math.sqrt(6)).T)
    # Energy measurement
    if bpf == True:
        # BPF only
        W = 1
    else:
        W = (H * np.conj(H)) / np.sum(FF*np.conj(FF), 1).reshape(-1, 1)
    # band limitation
    FMask = (freq >= LPF)&(freq <= HPF)
    FMask = FMask + FMask[::-1]
    W = W*FMask.reshape(-1, 1)
    # Weighting
    Ff = np.multiply(FF, (np.tile(W, [1, 3])))
    # temporal de-normalization
    C = np.array([(np.average(C_rgb, axis=0)),]*(L)) * np.abs(np.fft.ifft(Ff, axis=0)+1)
    return C

class RppgEstimator:

    def __init__(self, pbv_static=PBV_STATIC, pbv_update=PBV_UPDATE):
        self.pbv_static = np.array(pbv_static)
        self.pbv_update = np.array(pbv_update)
        self.last_hr = None

        # deactivates runtime warning if div by zero
        np.seterr(divide='ignore')

    def _sliceImageIntoNEqualParts(self, image, n):

        # get crops length for x and y axis
        n_x, n_y = image.shape[0] // n, image.shape[1] // n
        # get rid of edge pixels such that crops are all equally sized
        image = image[0:n_x * n, 0:n_y * n, :]

        crops = [image[x:x + n_x, y:y + n_y, :] for x in range(0, image.shape[0], n_x) for y in range(0, image.shape[1], n_y)]

        crops = np.stack(crops)

        return crops

    def estimateSpO2 (self, spat_mean_rgb, ror_parameter = ROR_PARAMETERS, fs = 30, moving_avg_window_size = ROR_DC_MVG_AVG_WINDOWS_SIZE):

        red_signal = spat_mean_rgb[:,0]
        green_signal = spat_mean_rgb[:, 1]
        blue_signal =  spat_mean_rgb[:,2]

        red_signal = butter_bandpass(red_signal, 0.7, 5, fs)
        blue_signal = butter_bandpass(blue_signal, 0.7, 5, fs)


        #moving average = dc component
        dc_red = np.ma.average(red_signal)
        dc_blue =  np.ma.average(blue_signal)

        min_peak_distance = fs / MAX_HEART_HZ

        #get ac componenmts
        ac_red = peak_to_through(red_signal, min_peak_distance)
        ac_blue = peak_to_through(blue_signal, min_peak_distance)

        a, b = ror_parameter[0], ror_parameter[1]
        ror = (ac_red/dc_red) / (ac_blue/ dc_blue)
        spo2 = a + b * ror

        return spo2, ror

    def estimateSpO2APBV(self, spat_mean_rgb_sliced, spo2_method="apbv", fs=30):

        #mean_signal_region = self.
        n_spo2_levels = len(PBV_TO_CHECK)
        n_subregions = spat_mean_rgb_sliced.shape[1]
        n_frames = spat_mean_rgb_sliced.shape[0]
        pulse_signals_spo2_levels_subregions = np.zeros((n_frames, n_spo2_levels, n_subregions) )
        snr_spo2_levels_subregions = np.zeros((n_spo2_levels,n_subregions,1))

        #iterate over each spo2 level and each sub region and get bvp signal.
        for spo2_level in range (0,n_spo2_levels):
            for subregion in range(0,n_subregions):
                #color signal of subroi
                mean_signal = spat_mean_rgb_sliced[:, subregion, :]
                mean_signal = np.expand_dims(mean_signal, axis=1)
                #apbv method
                # get pulse signal from one of the methods
                if spo2_method == 'pbv':
                    pulse_signal = self._pbv(mean_signal)
                elif spo2_method == 'pos':
                    pulse_signal = self._pos(mean_signal, fs)
                elif spo2_method == 'pos2':
                    pulse_signal = self._pos2(mean_signal, filter=True)
                elif spo2_method == 'chrom':
                    pulse_signal = self._chrom(mean_signal)
                elif spo2_method == 'apbv':
                    pulse_signal = self._apbv(mean_signal)

                #TODO: bandpass signal?
                pulse_signal = butter_bandpass(pulse_signal, MIN_HEART_HZ, MAX_HEART_HZ, fs, order=6)
                pulse_signals_spo2_levels_subregions [:, spo2_level, subregion] = np.squeeze(pulse_signal)

                fft, freq, _ =self.getFFTSpectrumMax(pulse_signal, fs)
                snr = signaltonoise(pulse_signal)
                snr_spo2_levels_subregions[spo2_level, subregion] = snr


        highest_quality_signal_idx = np.unravel_index(snr_spo2_levels_subregions.argmax(), snr_spo2_levels_subregions.shape)[0]
        spo2 = PBV_TO_CHECK[highest_quality_signal_idx]
        """import matplotlib.pyplot as plt
        plt.plot(pulse_signal)
        plt.show()"""
        return spo2

    def estimateHR(self, signal, fs):

        _, _, max_hr_hz =  self.getFFTSpectrumMax(signal, fs)
        hr = int(max_hr_hz * 60)

        return hr

    #returns fft, frequencies and max
    def getFFTSpectrumMax (self, signal, fs):
        delta = 1/fs
        n = len(signal)
        fminind = int(MIN_HZ * n * delta)
        fmaxind = int(MAX_HZ * n * delta)
        hr_fminind = int(MIN_HEART_HZ * n * delta) - fminind
        hr_fmaxind = int(MAX_HEART_HZ * n * delta) - fminind
        fminind += 1
        hr_fminind += 1
        if hr_fminind<0: hr_fminind=0

        spectrum = np.abs(np.fft.fft(signal))
        freq = np.fft.fftfreq(spectrum.shape[0], d=delta)[fminind:fmaxind + 1]
        spectrum_filtered = spectrum[fminind:fmaxind + 1]
        freq_idx = spectrum_filtered[hr_fminind:hr_fmaxind].argmax() + hr_fminind
        max_hz = freq[freq_idx]

        return spectrum_filtered, freq, max_hz


    # slices roi and brings it in correct shape
    # input roi in shape [frame, x, y, chan] or [x,y,chan]
    def _sliceIntoSubROIs(self, rois, n_roi_slices, n_channels = 3):
        # expands dimension to [1, x, y, chan]
        #if np.ndim(rois) == 3:
        #    rois = np.expand_dims(rois, axis=0)

        # get median shape of roi
        x_shape, y_shape = np.array([]), np.array([])
        for roi in rois:
            x_shape = np.append(x_shape, roi.shape[0])
            y_shape = np.append(y_shape, roi.shape[1])

        # rois must be equally sized, therefore we take the median dimensions
        x_max_shape = int(np.median(x_shape))
        y_max_shape = int(np.median(y_shape))

        n_frames = len(rois)
        rois_reshaped = np.zeros((n_frames, n_roi_slices ** 2,
                                  x_max_shape // n_roi_slices, y_max_shape // n_roi_slices,
                                  n_channels))

        for roi, i in zip(rois, range(n_frames)):
            # rescale roi to median shape
            roi_resized = cv2.resize(roi, dsize=(y_max_shape, x_max_shape), interpolation=cv2.INTER_CUBIC)
            # slice into sub rois
            roi_slice = self._sliceImageIntoNEqualParts(roi_resized, n_roi_slices)
            rois_reshaped[i, :, :, :, :] = roi_slice

        return rois_reshaped

    #rois has shape [frame, crop, x, y, channel]
    #return the mean rgb, and the mean rgb for each slice
    def getMeanRGBSignals(self, input_roi, n_roi_slices=1, n_channels=3, fs=30):
        #processing a vector us only used when we use a vector as an input in case of skin segmentation
        if np.ndim(input_roi[0])==2:
            spat_mean_rgb_map = map(lambda x: np.mean(x, axis=0), input_roi)
            spat_mean_rgb = np.stack(list(spat_mean_rgb_map))
            spat_mean_rgb = np.expand_dims(spat_mean_rgb, axis=1)
            spat_mean_rgb_sliced = []


        else:
            #spatial average over whole roi
            spat_mean_rgb = np.zeros ( (len(input_roi) ,1, 3))
            for i,roi in enumerate(input_roi):
                spat_mean_rgb[i,:,:] = np.mean(roi, axis = (0,1))

            #spatial avergae over subroi
            rois_sliced = self._sliceIntoSubROIs(input_roi, n_roi_slices=n_roi_slices, n_channels = n_channels)
            spat_mean_rgb_sliced = np.mean(rois_sliced, axis=(2, 3))

        #detrend and bp filter
        #spat_mean_rgb = butter_bandpass(spat_mean_rgb, 0.7, 3.5, fs, order=6)

        return spat_mean_rgb, spat_mean_rgb_sliced

    #spat_mean_rgb has shape [frame, crop, channel]
    def estimateVitalSignsInThread(self, spat_mean_rgb, spat_mean_rgb_sliced, fs=30, method='pos', spo2_method="apbv"):
        vital_sign_thread = threading.Thread(target=self.estimateVitalSigns,args=(spat_mean_rgb, spat_mean_rgb_sliced))
        vital_sign_thread.daemon =True

        vital_sign_thread

    def estimateVitalSigns(self, spat_mean_rgb, spat_mean_rgb_sliced, fs=30, method='pos', spo2_method="apbv"):
        mean_signal = np.squeeze(spat_mean_rgb)

        #get pulse signal from one of the methods
        if method == 'pbv':
            pulse_signal = self._pbv(mean_signal)
        elif method=='pos':
            pulse_signal = self._pos(mean_signal, fs)
        elif method =='pos2':
            pulse_signal = self._pos2(mean_signal, filter=True)
        elif method == 'chrom':
            pulse_signal=self._chrom(mean_signal)
        elif method == 'apbv':
            pulse_signal = self._apbv(mean_signal)

        #bandpass signal for better hr estimation
        pulse_signal = butter_bandpass(pulse_signal, MIN_HEART_HZ, MAX_HEART_HZ, fs, order=6)

        #return pulse_signal

        # #estimate hr
        #commenting hr_rate_measurement for getting fps for p2p

        hr = 0  
        hr = self.estimateHR(pulse_signal, fs)
        hrv = 0
        spo2 = 0 
        ror = 0 
        spo2, ror = self.estimateSpO2(mean_signal, fs=fs)

        result={'hr': hr, 'spo2': spo2, 'resp_rate': 0, "hrv":0 , "ror": ror}

        return result , pulse_signal 

    def _apbv (self, spat_mean_rgb):
        spat_mean_rgb /= np.mean(spat_mean_rgb, axis=0)
        spat_mean_rgb -= 1.0

        c_n = spat_mean_rgb.reshape((-1, 3)).T

        q_inv = np.linalg.inv(np.matmul(c_n, c_n.T))
        sp02_val = 1
        w_pbv = np.matmul([self.pbv_static + (1 - sp02_val) * self.pbv_update],q_inv)
        w_pbv /= np.linalg.norm(w_pbv)
        patch_pulse = np.matmul(w_pbv, c_n).T

        return patch_pulse

    ########################
    # The below mentioned methods are adapted from the following git:
    # https://github.com/phuselab/pyVHR which belongs to the following paper:
    # G.Boccignone, D.Conte, V.Cuculo, A.D’Amelio, G.Grossi and R.Lanzarotti,
    #"An Open Framework for Remote-PPG Methods and their Assessment," in IEEE Access, doi: 10.1109 / ACCESS.2020.3040936.
    def _pos (self, spat_mean_rgb, fs):
        wlen = int(1.6*fs)

        projection = np.array([[0, 1, -1], [-2, 1, 1]])
        spat_mean_rgb = spat_mean_rgb.T
        # Initialize (1)
        h = np.zeros(spat_mean_rgb.shape[1])
        for n in range(spat_mean_rgb.shape[1]):
            # Start index of sliding window (4)
            m = n - wlen + 1
            if m >= 0:
                # Temporal normalization (5)
                cn = spat_mean_rgb[:, m:(n + 1)]
                #cn = np.dot(self.__get_normalization_matrix(cn), cn)
                cn = cn / np.average (cn, axis=0)
                # Projection (6)
                s = np.dot(projection, cn)
                # alpha Tuning (7)
                hn = np.add(s[0, :], np.std(s[0, :]) / np.std(s[1, :]) * s[1, :])
                # Overlap-adding (8)
                h[m:(n + 1)] = np.add(h[m:(n + 1)], hn - np.mean(hn))

        return h

    def _pos2 (self, rgb_components, WinSec=1.6, LPF=0.7, HPF=2.5, fs=30, filter = False):
        """
        POS method
        WinSec :was a 32 frame window with 20 fps camera
        (i) L = 32 (1.6 s), B = [3,6]
        (ii) L = 64 (3.2 s), B = [4,12]
        (iii) L = 128 (6.4 s), B = [6,24]
        (iv) L = 256 (12.8 s), B = [10,50]
        (v) L = 512 (25.6 s), B = [18,100]
        """
        # 初期化
        N = rgb_components.shape[0]
        H = np.zeros(N)
        l = math.ceil(WinSec*fs)


        # loop from first to last frame
        for t in range(N-l+1):
            # spatical averagining
            C = rgb_components[t:t+l, :]
            if filter:
                C = cdf_filter(C, LPF, HPF, fs=fs,bpf=True)
                Cn = C / np.average(C, axis=0)
            Cn = C/np.average(C, axis=0)

            # projection (orthogonal to 1)
            S = np.dot(Cn, np.array([[0,1,-1],[-2,1,1]]).T)
            # alpha tuning
            P = np.dot(S, np.array([[1, np.std(S[:,0]) / np.std(S[:,1])]]).T)
            # overlap-adding
            H[t:t+l] = H[t:t+l] + (np.ravel(P)-np.mean(P))/np.std(P)

        return H



    # Compute a diagonal matrix n such that the mean of n*x is a vector of ones
    # this method is specificaly for the pos the method
    def __get_normalization_matrix(self, x):
        d = 0 if (len(x.shape) < 2) else 1
        m = np.mean(x, d)
        n = np.array([[1 / m[i] if i == j and m[i] else 0 for i in range(len(m))] for j in range(len(m))])
        return n


    def _pbv (self, spat_mean_rgb):
        r_mean = spat_mean_rgb[:,0 ] / np.mean(spat_mean_rgb[:,0])
        g_mean = spat_mean_rgb[:,1] / np.mean(spat_mean_rgb[:,1])
        b_mean = spat_mean_rgb[:,2] / np.mean(spat_mean_rgb[:,2])

        pbv_n = np.array([np.std(r_mean), np.std(g_mean), np.std(b_mean)])
        pbv_d = np.sqrt(np.var(r_mean) + np.var(g_mean) + np.var(b_mean))
        pbv = pbv_n / pbv_d

        C = np.array([r_mean, g_mean, b_mean])
        Q = np.matmul(C, np.transpose(C))
        W = np.linalg.solve(Q, pbv)

        bvp = np.matmul(C.T, W) / (np.matmul(pbv.T, W))

        return bvp



    def _chrom (self, spat_mean_rgb):
        spat_mean_rgb /= np.mean(spat_mean_rgb, axis=0)
        spat_mean_rgb -= 1.0

        Xcomp = 3 * spat_mean_rgb[:,0] - 2 * spat_mean_rgb[:,1]
        Ycomp = (1.5 * spat_mean_rgb[:,0]) + spat_mean_rgb[:,1] - (1.5 * spat_mean_rgb[:,2])

        # standard deviations
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)

        alpha = sX / sY

        # -- rPPG signal
        bvp = Xcomp - alpha * Ycomp

        return bvp
