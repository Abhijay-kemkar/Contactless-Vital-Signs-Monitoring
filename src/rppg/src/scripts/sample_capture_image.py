#!/usr/bin/env python3

# PKG = 'blimp'
# import roslib; roslib.load_manifest(PKG)

import rospy
from rospy_tutorials.msg import Floats
from blimp.msg import RGB
from rospy.numpy_msg import numpy_msg

import numpy

# from rppg.rppg import *
from numpy import genfromtxt
import numpy as np
import glob
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from sklearn.metrics import *
# from math import *
# import time
# import pdb

file_directory = "/home/naitik/Work/MIT/raw_rgb_signals/*"
file_extension = ".csv"

input_paths = glob.glob(file_directory+file_extension)

def talker() :
    pub = rospy.Publisher('floats', numpy_msg(RGB), queue_size=10)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(30)

    while not rospy.is_shutdown() :
        my_data = genfromtxt(input_paths[0], delimiter=',')
        #print(input_paths[0])
        # subject_num = input_paths[0].replace("/home/naitik/Work/MIT/raw_rgb_signals/subject","")
        # subject_num = subject_num.replace(".csv","")
        #X_input_samples.append(int(subject_num))
        my_data = my_data[1:]
        # X = np.zeros((my_data.shape[0],3))
        # y = np.zeros((my_data.shape[0],1))
        for i in range(len(my_data)):
            m_rgb =  numpy.array([my_data[i][0],my_data[i][1],my_data[i][2]] , dtype = numpy.float32)
            #print(m_rgb)
            pub.publish(m_rgb)
            r.sleep()
        r.sleep()

if __name__ == '__main__' :
    talker()
