import os
import sys
import cv2
import numpy as np
import json
from .ProjectiveObject import ProjectiveObject
from . import lib

class DepthCamera(ProjectiveObject):
    def __init__(self, param_path = ""):
        super().__init__(param_path)

        grid = np.indices((self.height, self.width))
        v = np.array([grid[0].ravel()])
        u = np.array([grid[1].ravel()])
        ones = np.ones(v.shape)
        uv_indices_homo_points = np.concatenate( (np.concatenate((u,v),axis=0),ones),axis=0 )
        self.uv_ray_list = np.dot(np.linalg.inv(self.intrinsic), uv_indices_homo_points)

        #print(uv_indices_homo_points.shape)
    
    def getPointCloudFromDepthImage(self, depth_img):
        depth_3ch = depth_img.ravel()/1000.0
        pointcloud = self.uv_ray_list * depth_3ch
        return pointcloud

    def getPointCloudFromDepthImageFile(self, depth_img_path):
        depth_img = lib.imreadKorean(depth_img_path)
        return self.getPointCloudFromDepthImage(depth_img)

if __name__=="__main__":
    dc = DepthCamera("depth_intrin.json")
    print(dc.intrinsic)
    print(dc.distortion)
    print(dc.width)
    print(dc.height)
    image = lib.imreadKorean("depth.png")
    #print(image.shape)
    pointcloud = dc.getPointCloudFromDepthImage(image)