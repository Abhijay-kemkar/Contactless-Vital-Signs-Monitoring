import jetson.utils
import pdb

display = jetson.utils.glDisplay()
import pickle
from rppg.rppg import RppgEstimator
import matplotlib.pyplot as plt

roi_buffer = pickle.load(open("results/test_roi_buffer.p", "rb"))
rppg = RppgEstimator()

mean_rgb, mean_rgb_sliced = rppg.getMeanRGBSignals(roi_buffer, n_roi_slices=1, n_channels=3)
vital_signs, pulse_signal = rppg.estimateVitalSignsInThread(mean_rgb, mean_rgb_sliced, fs=32, spo2_method="pos")
print(vital_signs)

plt.plot(pulse_signal)
plt.show()
pdb.set_trace()
"""for frame in roi_buffer:
    img_cuda = jetson.utils.cudaFromNumpy(frame)
    display.RenderOnce(img_cuda)"""