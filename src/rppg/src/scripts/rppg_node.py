#!/usr/bin/env python

# PKG = 'blimp'
# import roslib ; roslib.load_manifest(PKG)

#!/usr/bin/env python

import rospy
from rospy_tutorials.msg import Floats

def callback(data) :
    print(data.data)

def listener():
    rospy.init_node('listener')
    rospy.Subscriber("floats", Floats, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
