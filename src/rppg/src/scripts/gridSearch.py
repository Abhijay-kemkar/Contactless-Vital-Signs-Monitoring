import os

strides = [30, 60, 90, 150, 300]
sizes = [150, 300, 450, 600, 900]
hr_reset_rates = [0,2,3,4]

for stride in strides:
    for size in sizes:
        for hr_reset in hr_reset_rates:
            args = " --sw_stride {:}  --buffer_size {:} --hr_reset {:} -grid_search".format(stride, size, hr_reset)
            os.system('sudo python3 main_video.py' + args)
