import os
import sys
import cv2
import numpy as np
import json
from calipy.DepthCamera import DepthCamera
from calipy.ColorCamera import ColorCamera
from calipy.Transform import Transform
import calipy.lib

class RGBDCamera():
    def __init__(self, depth_param_path, color_param_path, tranform_path):
        self.depth_camera =  DepthCamera(depth_param_path)
        self.color_camera =  ColorCamera(color_param_path)
        self.transform = Transform(tranform_path)
        self.transform_inv = self.transform.inv()

    def getPointcloudTexture(self, pointcloud_list, tex_img):
        #print(pointcloud_list)
        tex_img = cv2.cvtColor(tex_img,cv2.COLOR_BGR2RGB)
        arrange_points = self.transform.translate(pointcloud_list)
        arrange_points = np.dot(self.color_camera.intrinsic, arrange_points)
        tex_data = arrange_points / arrange_points[2,:]
        x = np.reshape(tex_data[0,:],(self.depth_camera.height,self.depth_camera.width))
        y = np.reshape(tex_data[1,:],(self.depth_camera.height,self.depth_camera.width))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        tex_img = cv2.remap(tex_img,x,y,interpolation=cv2.INTER_LINEAR)
        tex_img = np.transpose(np.reshape(tex_img, (-1,3)))
        return tex_img

    def getPointcloudTextureFromImageFile(self, pointcloud_list, tex_img_path):
        tex_img = calipy.lib.imreadKorean(tex_img_path)
        return self.getPointcloudTexture(pointcloud_list, tex_img)
    
    def translatePointsToColorCoordinate(self, pointcloud):
        return self.transform.translate(pointcloud)

    def translatePointsToDepthCoordinate(self, pointcloud):
        return self.transform_inv.translate(pointcloud)
