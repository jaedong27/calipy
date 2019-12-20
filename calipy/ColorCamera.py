import os
import sys
import cv2
import numpy as np
import json
from calipy.ProjectiveObject import ProjectiveObject
from calipy import imglib
import pptk

class ColorCamera(ProjectiveObject):
    def __init__(self, param_path = ""):
        super().loadJson(param_path)
        
if __name__=="__main__":
    dc = ColorCamera("color_intrin.json")