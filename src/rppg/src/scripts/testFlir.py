from inputstream.flir.blackfly import Blackfly
import PySpin
import pdb

# Retrieve singleton reference to system object
system = PySpin.System.GetInstance()
# init camera
cam = Blackfly(system)

cam.beginAcquistion(0)

i =0
frames_buffer=[]
while True:
    img = cam.getFrame()
    frames_buffer.append(img)
    pdb.set_trace()
    print(len(frames_buffer))
