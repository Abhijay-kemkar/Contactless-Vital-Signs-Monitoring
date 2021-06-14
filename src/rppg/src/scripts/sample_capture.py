#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32 , Float32

rospy.init_node('sample_capture_node')

pub = rospy.Publisher('meanrgb', Int32)

rate = rospy.Rate(2)

count = 0

while not rospy.is_shutdown():
	pub.publish(count)
	count += 1
	rate.sleep()





#changes from the Abhijay's code :
# 1. in this node we are only publishing 1 value. Where there have to be 3 values. So we need to publish an array.
# 2. Dataype needs to be float32 / float64 instead of int. As the mean values are not whole numbers.
# 3. For the point 1, we need to create a custom message.
