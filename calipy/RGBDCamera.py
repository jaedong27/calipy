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

    def reprojectionDepthImagePointsOnColorCoordinate(self, pointcloud):
        pointcloud_on_color = self.transform.move(pointcloud)
        #print(points)
        uvs = np.dot(self.color_camera.intrinsic, pointcloud_on_color)
        uvs = uvs[:2,:] / uvs[2,:]
        #print(uvs)
        uvs = (uvs+0.5).astype(np.int)
        #print(uvs)
        #(points[,:] > 0)
        valid_points = np.logical_and(np.logical_and(uvs[0,:] > 0, uvs[0,:] < self.color_camera.width), np.logical_and(uvs[1,:] > 0, uvs[1,:] < self.color_camera.height) )
        depth_on_color = np.zeros((self.color_camera.height,self.color_camera.width,3))
        uvs = np.flip(uvs,axis=0)
        depth_on_color[tuple(uvs[:,valid_points].tolist())] = np.transpose(tuple(pointcloud[:,valid_points]))
        # result_depth_on_color = depth_on_color.copy()
        # valid_value = depth_on_color[:,:,2] > 0
        # for v in range(2, depth_on_color.shape[0]-2):
        #     for u in range(2,depth_on_color.shape[1]-2):
        #         if valid_value[v,u]:
        #             continue
        #         region_mask = valid_value[v:v+5,u:u+5]
        #         sum_value = np.sum(region_mask)
        #         if sum_value == 0:
        #             continue
        #         #print(sum_value)
        #         result_depth_on_color[v,u,0] = np.sum(depth_on_color[v:v+5,u:u+5,0]) / sum_value
        #         result_depth_on_color[v,u,1] = np.sum(depth_on_color[v:v+5,u:u+5,1]) / sum_value
        #         result_depth_on_color[v,u,2] = np.sum(depth_on_color[v:v+5,u:u+5,2]) / sum_value
        #         #print(region_value)
        
        return depth_on_color