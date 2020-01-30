import os
import sys
import cv2
import numpy as np
import json
from calipy.ProjectiveObject import ProjectiveObject

class Projector(ProjectiveObject):
    def __init__(self, param_path = ""):
        super().__init__(param_path)
        
    def projectImageOnPointCloud(self, points_vec, proj_img):
        arrange_points = np.dot(self.intrinsic, points_vec)
        uv_data = arrange_points / arrange_points[2,:]
        x = uv_data[0,:].astype(np.float32)
        y = uv_data[1,:].astype(np.float32)
        x_valid_data = np.logical_and(x >= 0, x < self.width)
        y_valid_data = np.logical_and(y >= 0, y < self.height)
        valid_data = np.logical_and(x_valid_data, y_valid_data)

        valid_uv_data = uv_data[:,valid_data].astype(np.float32)
        valid_color_list = proj_img[valid_uv_data[1,:].astype(np.int), valid_uv_data[0,:].astype(np.int)]
        
        color_list = np.zeros((points_vec.shape[1],4))
        color_list[valid_data,0:3] = valid_color_list[:,0:3]
        color_list[valid_data,3:4] = 255

        return color_list, valid_data

if __name__=="__main__":
    projector = Projector("projector.json")
    print(projector.intrinsic)