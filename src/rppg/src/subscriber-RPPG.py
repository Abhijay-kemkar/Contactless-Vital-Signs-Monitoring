#!/usr/bin/env python3

import rospy
import numpy
from rospy_tutorials.msg import Floats
from rppg.msg import RGB
from rospy.numpy_msg import numpy_msg
from scripts.rppg.rppg import *

mrgb = []
rppg = RppgEstimator()

def callback(msg):

	mrgb.append(msg.m_rgb)

	if (len(mrgb)==600):

		X = numpy.array(list1)

		for i in range(30) :
			del mrgb[0]

		result , pulse_signal = rppg.estimateVitalSigns(X, None, fs=30, method='pos', spo2_method= None)
		print(result['hr'])
		print("")

def listener() :

	rospy.init_node('RPPG')

	sub = rospy.Subscriber('roicoordinates', numpy_msg(RGB), callback)

	rospy.spin()

if __name__ == '__main__':
	listener()
