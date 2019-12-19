import cv2
import numpy as np
import json
import os

class ProjectiveObject():
    def __init__(self, param_path = ""):
        if self.loadJson(param_path) == False:
            #rms = 0.137250
            fx = 1382.0
            fy = 1382.0
            cx = 959.241044
            cy = 522.888753
            #0.120684  -0.236034  -0.001411  -0.003282
            k1 = 0.089712
            k2 = -0.157344
            p1 = -0.000022
            p2 = -0.000450
            distortion = np.array([k1, k2, p1, p2])
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            self.height = 1080
            self.width = 1920

            self.intrinsic = intrinsic
            self.distortion = distortion
            self.intrinsic_inv = np.linalg.inv(self.intrinsic)
            print("ProjectiveObject Load Fail : ", param_path)
        else:
            print("ProjectiveObject Load Success : ", param_path)
        self.setRemapParam()

        self.img = np.zeros((1080,1920,3))

    def setRemapParam(self):
        self.mtx = self.intrinsic
        self.dist = self.distortion
        w = self.width
        h = self.height
        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1)
        self.mapx,self.mapy = cv2.initUndistortRectifyMap(self.mtx,self.dist,None,self.newcameramtx,(w,h),5)

    def loadJson(self, path):
        if not os.path.exists(path):
            print("No File, Load Fail:", path)
            return False

        with open(path) as f:
            data = json.load(f)

        if 'width' not in data or 'height' not in data:
            print("File doesn't have Size Param(width or height)")
            return False 
        self.width = data["width"]
        self.height = data["height"]

        if 'intrinsic' in data:
            self.intrinsic = np.array(data["intrinsic"])
        elif 'ppx' in data:
            fx = np.array(data["fx"])
            fy = np.array(data["fy"])
            cx = np.array(data["ppx"])
            cy = np.array(data["ppy"])
            self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            #0.120684  -0.236034  -0.001411  -0.003282
        else:
            print("Camera Intrinsic Load Fail")
            return False

        if 'coeffs' in data:
            self.distortion = np.array(data["coeffs"])
        elif 'distortion' in data:
            self.distortion = np.array(data["distortion"])
            #0.120684  -0.236034  -0.001411  -0.003282
        else:
            self.distortion = np.array([0,0,0,0])
            print("Projective Object set default distortion : ", self.distortion)

        self.intrinsic_inv = np.linalg.inv(self.intrinsic)
        return True

    def saveJson(self, path):
        data = {}
        data["intrinsic"] = self.intrinsic.tolist()
        data["distortion"] = self.distortion.tolist()
        data["width"] = self.width
        data["height"] = self.height

        with open(path, 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)

    def solveDistortion(self, img):
        undistor_img = cv2.remap(img,self.mapx,self.mapy,cv2.INTER_LINEAR)
        return undistor_img

    def removePointcloudOutsideFrame(self, pointcloud_list, tex = None):
        uv_points_at_camera = np.dot(self.intrinsic, pointcloud_list)
        uv_points_at_camera = uv_points_at_camera[0:2,:] / uv_points_at_camera[2,:]
        
        valid_range_x = np.logical_and(uv_points_at_camera[0,:] > 0, uv_points_at_camera[0,:] < self.width - 1)
        valid_range_y = np.logical_and(uv_points_at_camera[1,:] > 0, uv_points_at_camera[1,:] < self.height - 1)
        valid_range = np.logical_and(valid_range_x, valid_range_y)

        arrange_pointcloud_list = pointcloud_list[:, valid_range]
        uv_points_at_camera = uv_points_at_camera[:, valid_range]
        if tex is None:
            arrange_tex = None
        else:
            arrange_tex = tex[:, valid_range]
        return arrange_pointcloud_list, uv_points_at_camera, arrange_tex

    def projectPointcloud(self, pointcloud_list, uv_list, tex = None):
        depth_img = np.zeros((int(self.height), int(self.width)))
        depth_img[np.int_(uv_list[1,:]),np.int_(uv_list[0,:])] = pointcloud_list[2,:]

        color_img = np.zeros((int(self.height), int(self.width),3))
        if tex is None:
            color_img[np.int_(uv_list[1,:]),np.int_(uv_list[0,:])] = 255*np.ones((1,3))
        else:
            color_img[np.int_(uv_list[1,:]),np.int_(uv_list[0,:])] = np.transpose(tex)
        return depth_img, color_img

    def getProjectorUV(self, points_list):
        uv_list = np.dot(self.intrinsic, points_list)
        uv_list = uv_list / uv_list[2,:]
        return uv_list

    def getHomographyFromUVOnProjector(self, uv_list):
        #print(uv_list) # left-top, right-top, left-bot, right-bot
        uv_list[0,:] = uv_list[0,:] / self.width
        uv_list[1,:] = uv_list[1,:] / self.height
        origin_point = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])  ###
        H = cv2.findHomography(origin_point, np.transpose(uv_list))
        #print("A", H)
        return H[0]

if __name__=="__main__":
    po = ProjectiveObject("depth_intrin.json")
    print(po.intrinsic)
    print(po.distortion)
    print(po.width)
    print(po.height)