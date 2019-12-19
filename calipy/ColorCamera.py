import os
import sys
import cv2
import numpy as np
import json
from ProjectiveObject import ProjectiveObject
sys.path.append("C:/Users/jaedong/Desktop/VRTProjectionIntervalDetection/Util")
import img
import pptk

class ColorCamera(ProjectiveObject):
    def __init__(self, param_path = ""):
        super().loadJson(param_path)
        
if __name__=="__main__":
    dc = ColorCamera("color_intrin.json")