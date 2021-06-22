#!/usr/bin/env python3

import rospy
import numpy
import datetime
import time
import cv2
import argparse
import imutils
from threading import Thread
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def CaptureImage() :

    prev_frame_time = 0
    new_frame_time = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    pub = rospy.Publisher('video_pub_py', Image, queue_size=10)

    rospy.init_node('CaptureImage', anonymous=True)

    rate = rospy.Rate(30)

    vs = WebcamVideoStream(src=0).start()

    br = CvBridge()

    while not rospy.is_shutdown():

        frame = vs.read()
        pub.publish(br.cv2_to_imgmsg(frame))
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rate.sleep()

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__' :
    CaptureImage()
