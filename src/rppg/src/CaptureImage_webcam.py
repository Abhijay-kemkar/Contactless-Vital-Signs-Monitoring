#!/usr/bin/env python3

import cv2
import rospy
from rospy_tutorials.msg import Floats
from rppg.msg import RGB
from rospy.numpy_msg import numpy_msg
import numpy

def CaptureImage() :
    rospy.init_node('CaptureImage', anonymous=True)
    
    pub = rospy.Publisher('image', numpy_msg(RGB), queue_size=10)

    rate = rospy.Rate(30)
    
    while not rospy.is_shutdown():
        vid = cv2.VideoCapture(0) #TODO set 30 fps frame
        
        while(True):
            ret, frame = vid.read()
            image =  numpy.array(frame, dtype = numpy.float32)
            pub.publish(image)
            print(image.shape)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__' :
    CaptureImage()
