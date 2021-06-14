from inputstream.input import InputStream
import cv2
import os

#loads all dataset from a folder an evaluates them
class VideoInput (InputStream, video_extensions = ['mp4', 'mov', 'wmv', 'avi']):

    def __init__(self, path):
        self.path = path

        f = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            #checks if filenames contains a valid video extension
            if any(ext in filenames for ext in video_extensions):
                f.append(filenames)

        self.filenames = f
