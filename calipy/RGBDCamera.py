import os
import sys
import cv2
import numpy as np
import json
from calipy.DepthCamera import DepthCamera
from calipy.ColorCamera import ColorCamera
from calipy.Transform import Transform
import calipy.lib
import pptk

class RGBDCamera():
    def __init__(self, depth_param_path, color_param_path, tranform_path):
        self.depth_camera =  DepthCamera(depth_param_path)
        self.color_camera =  ColorCamera(color_param_path)
        self.transform =  Transform(tranform_path)

    def getPointcloudTexture(self, pointcloud_list, tex_img):
        #print(pointcloud_list)
        tex_img = cv2.cvtColor(tex_img,cv2.COLOR_BGR2RGB)
        arrange_points = self.transform.translate(pointcloud_list)
        #print(arrange_points)
        arrange_points = np.dot(self.color_camera.intrinsic, arrange_points)
        tex_data = arrange_points / arrange_points[2,:]
        #print(tex_data.shape, img.shape)
        x = np.reshape(tex_data[0,:],(self.depth_camera.height,self.depth_camera.width))
        y = np.reshape(tex_data[1,:],(self.depth_camera.height,self.depth_camera.width))
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        tex_img = cv2.remap(tex_img,x,y,interpolation=cv2.INTER_LINEAR)
        tex_img = np.transpose(np.reshape(tex_img, (-1,3)))
        return tex_img

    def getPointcloudTextureFromImageFile(self, pointcloud_list, tex_img_path):
        tex_img = lib.imreadKorean(tex_img_path)
        return self.getPointcloudTexture(pointcloud_list, tex_img)
        
if __name__=="__main__":
    rs_cam = RGBDCamera("depth_intrin.json", "color_intrin.json", "depth_to_color_extrin.json")
    depth_img = lib.imreadKorean("depth.png")
    color_img = lib.imreadKorean("color.png")
    # cv2.imshow("test", color_img)
    # cv2.waitKey(0)
    pointcloud = rs_cam.depth_camera.getPointCloudFromDepthImage(depth_img)
    tex_img = rs_cam.getPointcloudTexture(pointcloud, color_img)
    v = pptk.viewer(np.transpose(pointcloud), np.transpose(tex_img)/255.0)
    v.set(point_size=0.01)