#!/usr/bin/env python3

import rospy
from rospy_tutorials.msg import Floats
from blimp.msg import RGB
from rospy.numpy_msg import numpy_msg
import numpy
from rppg.rppg import *

list1 = []
rppg = RppgEstimator()

def callback(msg):
	#print(msg.m_rgb)
	list1.append(msg.m_rgb)
	#print(len(list1))

	if (len(list1)==600):

		X = numpy.array(list1)
		print(X)

		for i in range(30) :
			del list1[0]

		#print(X_new.shape)

		result , pulse_signal = rppg.estimateVitalSigns(X, None, fs=30, method='pos', spo2_method= None)
		print(result)
		print("")


def listener() :

	rospy.init_node('RPPG')

	sub = rospy.Subscriber('floats', numpy_msg(RGB), callback)


	# while not rospy.is_shutdown():
	#
	# 	print("yes")
	# 	print(len(list1))


	rospy.spin()

if __name__ == '__main__':
	listener()
