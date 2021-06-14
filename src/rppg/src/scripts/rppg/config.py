CHAN_TXT = ('rgb',)
CHAN_NUM_CHANS = (3,)

# IMX265
## halogen illumination
PBV_STATIC = [ 0.92,    1.0,    0.45]
PBV_UPDATE = [-0.0047, -0.0016, 0.0003]

## sunlight illumination
#PBV_STATIC = [ 0.42,    1.0,    0.49]
#PBV_UPDATE = [-0.0034, -0.0011, 0.0004]

PBV_TO_CHECK = [100, 95, 90, 85, 80, 75, 70]
# Minimum of 45 beats per minute.
MIN_HEART_HZ = 40.0 / 60.0
# Maximum of 100 beats per minute.
MAX_HEART_HZ = 150.0 / 60.0
# Minimum of 30 beats per minute in stored FFT.
MIN_HZ = 50.0 / 60.0
# Enough to look for the 2nd harmonic of heart rate.
MAX_HZ = 2 * MAX_HEART_HZ

#Ratio or Ratios fitted parameters
ROR_PARAMETERS = [97.28778046886782, 0.012651773276269863]
ROR_DC_MVG_AVG_WINDOWS_SIZE = 10 #in seconds

QUALITY_PRUNE_FACTOR = 1.0
