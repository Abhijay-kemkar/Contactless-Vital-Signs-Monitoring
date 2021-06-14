from numpy import genfromtxt
import numpy as np
import glob
import matplotlib.pyplot as plt


MIN_PIXEL_HZ = 0 / 60.0

MAX_PIXEL_HZ = 255 / 60.0

MIN_HZ = 30 / 60.0

MAX_HZ = 2 * MAX_PIXEL_HZ


def signaltonoise(a):
    #for now let us keep noise = 1
    noise = 1
    power = np.max(a)
    index = np.max(a).argmax()

    if power > 1000 :
        if index == 0 :
            noise = sum(a[1:])
        else :
            noise = sum(a[:index])
            noise += sum(a[index+1:])

    return power/noise

file_directory = "/home/naitik/Work/MIT/raw_rgb_signals/*"
file_extension = ".csv"

input_paths = glob.glob(file_directory+file_extension)

X_input_samples = []

snr_overall = []

for i in range(len(input_paths)):

    my_data = genfromtxt(input_paths[i], delimiter=',')

    snr_samplewise = []

    subject_num = input_paths[i].replace("/home/naitik/Work/MIT/raw_rgb_signals/subject","")
    subject_num = subject_num.replace(".csv","")

    X_input_samples.append(int(subject_num))

    my_data = my_data[1:]
    R = np.zeros((my_data.shape[0],1))
    G = np.zeros((my_data.shape[0],1))
    B = np.zeros((my_data.shape[0],1))
    y = np.zeros((my_data.shape[0],1))

    for i in range(my_data.shape[0]):
        R[i][0] = my_data[i][0]
        G[i][0] = my_data[i][1]
        B[i][0] = my_data[i][2]
        y[i][0] = my_data[i][3]


    sample_start = 0
    sample_end = 150

    slide = 15

    snr_R_samplewise = []
    snr_G_samplewise = []
    snr_B_samplewise = []

    while sample_end < len(R) :

        R_signal = R[sample_start:sample_end]
        G_signal = G[sample_start:sample_end]
        B_signal = B[sample_start:sample_end]

        R_av = sum(R_signal)/len(R_signal)
        G_av = sum(G_signal)/len(G_signal)
        B_av = sum(B_signal)/len(B_signal)

        R_signal = R_signal - R_av
        G_signal = G_signal - G_av
        B_signal = B_signal - B_av


        plt.plot(R_signal , color = 'red')
        #plt.plot(G_signal, color = 'green')
        #plt.plot(B_signal, color = 'blue')

        plt.show()

        R_spectrum = np.abs(np.fft.fft(R_signal , axis = 0))
        G_spectrum = np.abs(np.fft.fft(G_signal , axis = 0))
        B_spectrum = np.abs(np.fft.fft(B_signal , axis = 0))


        freq_R = np.fft.fftfreq(R_spectrum.shape[0], d=1/30)
        freq_G = np.fft.fftfreq(R_spectrum.shape[0], d=1/30)
        freq_B = np.fft.fftfreq(R_spectrum.shape[0], d=1/30)

        # freq_R = freq_R - min(freq_R)

        # R_spectrum = R_spectrum - R_spectrum[0]

        print(freq_R)

        plt.plot( freq_R, R_spectrum ,color = 'red')
        #plt.plot(G_spectrum[10:137], color = 'green')
        #plt.plot(B_spectrum[10:137], color = 'blue')

        plt.show()


        index = np.max(R_spectrum).argmax()

        # print(index)
        # print(R_spectrum[index])
        # print(R_spectrum[index+1])
        # print(R_spectrum[index+2])
        # mean_R = sum(R_spectrum[1:])/(len(R_spectrum)-1)
        # print(sum(R_spectrum[1:])/(len(R_spectrum))-1)
        # print(len(R_spectrum[R_spectrum[:] > mean_R]))

        snr_R = signaltonoise(R_spectrum)
        snr_G = signaltonoise(G_spectrum)
        snr_B = signaltonoise(B_spectrum)


        snr_R_samplewise.append(snr_R)
        snr_G_samplewise.append(snr_G)
        snr_B_samplewise.append(snr_B)

        sample_start += slide
        sample_end += slide

    snr_zipped = (sum(snr_R_samplewise)/len(snr_R_samplewise),sum(snr_G_samplewise)/len(snr_G_samplewise),sum(snr_B_samplewise)/len(snr_B_samplewise))

    snr_overall.append(snr_zipped)


X = X_input_samples
Y1 = snr_overall

zipped_lists = zip(X,Y1)

sorted_zipped_lists = sorted(zipped_lists)

X.clear()
Y1.clear()
R_final = []
G_final = []
B_final = []

for x, y in enumerate(sorted_zipped_lists) :
    X.append(y[0])
    R_final.append(y[1][0][0])
    G_final.append(y[1][1][0])
    B_final.append(y[1][2][0])

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.25, R_final, 0.25, label = 'red' , color = 'red')
plt.bar(X_axis , G_final, 0.25, label = 'green' , color = 'green')
plt.bar(X_axis + 0.25 , B_final, 0.25, label = 'blue' , color = 'blue')

plt.xticks(X_axis, X)
plt.xlabel("Subjects")
plt.ylabel("SNR")
plt.title("SNR")
plt.legend()
plt.show()
