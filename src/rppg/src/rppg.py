#!/usr/bin/env python3

import rospy
from rospy_tutorials.msg import Floats
from rppg.msg import RGB
from rospy.numpy_msg import numpy_msg
import numpy
from rppg.rppg import *

meanrgb = []
rppg = RppgEstimator()

def callback(msg):
	meanrgb.append(msg.m_rgb)

	if (len(meanrgb)==600):
		X = numpy.array(meanrgb)
		for i in range(30) :
			del meanrgb[0]

		result , pulse_signal = rppg.estimateVitalSigns(X, None, fs=30, method='pos', spo2_method= None)
		print(result)
		print("")

def listener() :

	rospy.init_node('RPPG')

	sub = rospy.Subscriber('meanrgb', numpy_msg(RGB), callback)

	rospy.spin()

if __name__ == '__main__':
	listener()
