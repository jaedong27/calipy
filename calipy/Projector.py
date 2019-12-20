import os
import sys
import cv2
import numpy as np
import json
from calipy.ProjectiveObject import ProjectiveObject
import pptk

class Projector(ProjectiveObject):
    def __init__(self, param_path = ""):
        super().__init__(param_path)
        
if __name__=="__main__":
    projector = Projector("projector.json")
    print(projector.intrinsic)