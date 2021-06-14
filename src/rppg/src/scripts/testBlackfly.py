import jetson.utils
display = jetson.utils.glDisplay()
screen_resolution = (display.GetWidth(), display.GetHeight())

from inputstream.flir.blackfly import Blackfly
from config import *
import keyboard
import cv2


cam = Blackfly()
try:

    cam.beginAcquistion(0, mode_SETWIDTH = MODE_SETWIDTH, mode_SETHEIGHT= MODE_SETHEIGHT, mode_OFFSETX=MODE_OFFSETX, mode_OFFSETY=MODE_OFFSETY)

    path = "results/distance_img/"
    i=0
    while True:
        frame = cam.read()
        if keyboard.is_pressed("s"):
            cv2.imwrite(path+"flir_{:}.png".format(i), frame)
            i +=1
        frame= cv2.resize(frame, screen_resolution)
        img_cuda = jetson.utils.cudaFromNumpy(frame)
        display.RenderOnce(img_cuda)

except Exception as e:
    print(e)
    cam.endAll()