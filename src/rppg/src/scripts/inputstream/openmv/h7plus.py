from inputstream.input import InputStream
from config import *

class omv_h7plus (InputStream):


    def __init__(self, system):

        self.focal_length = 2.8
        self.wave_length = {'blue': 460, 'green': 530, 'red': 625}
        self.sensitivy_threshold = 6  #TODO: tbd, not given by manufacturer

    def getCamList(self):
        return self.cam_list


    def beginAcquistion (self, count, mode_OFFSETX = MODE_OFFSETX, mode_OFFSETY = MODE_OFFSETY,
                         mode_SETWIDTH = MODE_SETWIDTH,  mode_SETHEIGHT = MODE_SETHEIGHT):



    def endAcquistion(self):

    def deInit(self):

    def releaseInstance(self):

    def getFrame (self, pixelFormat =  PySpin.PixelFormat_BGR8):


        return image_data

    def getFocalLength(self):
        return self.focal_length
