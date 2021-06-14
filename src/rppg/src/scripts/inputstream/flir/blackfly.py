import PySpin
from config import *
import sys
import traceback

from inputstream.input import InputStream


def print_device_info(nodemap):
    """
    This function prints the device information of the camera from the transport
    layer; please see NodeMapInfo example for more in-depth comments on printing
    device information from the nodemap.
    :param nodemap: Transport layer device nodemap.
    :type nodemap: INodeMap
    :returns: True if successful, False otherwise.
    :rtype: bool
    """

    print("*** DEVICE INFORMATION ***\n")

    try:
        result = True
        node_device_information = PySpin.CCategoryPtr(nodemap.GetNode("DeviceInformation"))

        if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
            features = node_device_information.GetFeatures()
            for feature in features:
                node_feature = PySpin.CValuePtr(feature)
                print("%s: %s" % (node_feature.GetName(),
                                  node_feature.ToString() if PySpin.IsReadable(node_feature) else "Node not readable"))

        else:
            print("Device control information not available.")

    except PySpin.SpinnakerException as ex:
        print(traceback.format_exc())
        return False

    return result


def configure_custom_image_settings(nodemap, mode,
                                    mode0_OFFSETX, mode0_OFFSETY, mode0_SETWIDTH, mode0_SETHEIGHT,
                                    mode1_OFFSETX, mode1_OFFSETY, mode1_SETWIDTH, mode1_SETHEIGHT):
    """
    Configures a number of settings on the camera including offsets  X and Y, width,
    height, and pixel format. These settings must be applied before BeginAcquisition()
    is called; otherwise, they will be read only. Also, it is important to note that
    settings are applied immediately. This means if you plan to reduce the width and
    move the x offset accordingly, you need to apply such changes in the appropriate order.
    :param nodemap: GenICam nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    # mode == 1, 1x1
    SETWIDTH = mode1_SETWIDTH
    SETHEIGHT = mode1_SETHEIGHT
    OFFSETX = mode1_OFFSETX
    OFFSETY = mode1_OFFSETY

    if mode == 0:
        SETWIDTH = mode0_SETWIDTH
        SETHEIGHT = mode0_SETHEIGHT
        OFFSETX = mode0_OFFSETX
        OFFSETY = mode0_OFFSETY

    node_width = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
    node_width.SetValue(SETWIDTH)

    node_height = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
    node_height.SetValue(SETHEIGHT)

    node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
    node_offset_x.SetValue(OFFSETX)

    node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
    node_offset_y.SetValue(OFFSETY)

    return True


class Blackfly (InputStream):

    def __init__(self):
        self.system = PySpin.System.GetInstance()

        self.focal_length = 12
        self.wave_length = {'blue': 460, 'green': 530, 'red': 625}
        self.sensitivy_threshold = 5.22


        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()

        num_cameras = self.cam_list.GetSize()

        print("Number of cameras detected: %d" % num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            self.cam_list.Clear()

            # Release system
            self.system.ReleaseInstance()

            print("Not enough flir cameras!")

            sys.exit()


        cam = self.cam_list.GetByIndex(0)

        print("Running example for camera %d..." % 0)



        """
            This function acts as the body of the example; please see NodeMapInfo example
            for more in-depth comments on setting up cameras.
            :param cam: Camera to run on.
            :type cam: CameraPtr
            :return: True if successful, False otherwise.
            :rtype: bool
            """
        try:
            result = True

            # Retrieve TL device nodemap and print device information
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            result &= print_device_info(nodemap_tldevice)

            # Initialize camera
            cam.Init()

            self.cam = cam

            #return cam

        except PySpin.SpinnakerException as ex:
            print(traceback.format_exc())


    def getCamList(self):
        return self.cam_list


    def beginAcquistion (self, count, mode_OFFSETX = MODE_OFFSETX, mode_OFFSETY = MODE_OFFSETY,
                         mode_SETWIDTH = MODE_SETWIDTH,  mode_SETHEIGHT = MODE_SETHEIGHT):


        # Retrieve GenICam nodemap
        nodemap = self.cam.GetNodeMap()
        configure_custom_image_settings(nodemap, count % 2, mode_OFFSETX, mode_OFFSETY, mode_SETWIDTH, mode_SETHEIGHT,
                                        mode_OFFSETX, mode_OFFSETY, mode_SETWIDTH, mode_SETHEIGHT)



        # get rgb not grey scale
        self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)

        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName("Continuous")
        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        self.cam.BeginAcquisition()

    def endAcquistion(self):
        self.cam.EndAcquisition()

    def deInit(self):
        self.cam.DeInit()

    #calls deInit, endAcquisition and releaseInstance
    def endAll(self):
        self.endAcquistion()
        self.deInit()

    def releaseInstance(self):
        self.system.ReleaseInstance()

    def read (self, pixelFormat =  PySpin.PixelFormat_BGR8):
        image_result = self.cam.GetNextImage()

        image_converted = image_result.Convert(pixelFormat)

        # Getting the image data as a numpy array
        image_data = image_converted.GetNDArray()

        return image_data

    def getFocalLength(self):
        return self.focal_length