import numpy as np
import cv2
import glob
import sys
sys.path.append("../")
import calipy


data_list_seed = "./CameraData/Back/*.png"
output_path = "./CameraData/Back/cam_intrinsic.json"

# data_list_seed = "./CameraData/Front/*.png"
# output_path = "./CameraData/Front/cam_intrinsic.json"

img_list = glob.glob(data_list_seed)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((3*4,3), np.float32)
objp[:,:2] = np.mgrid[0:3,0:4].T.reshape(-1,2) * 0.135

print(objp)
#exit()

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fname in img_list:
    print("Path: ", fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (3,4), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (3, 4), corners2, ret)
        cv2.imshow('img',img)
        
        ch = cv2.waitKey(0)
        if ch == ord('q'):
            exit()
        elif ch == ord('e'):
            break
        elif ch == ord(' '):
            print(fname, " -> Add")
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(fname, " -> Skip!!!")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(ret)
print(mtx)
print(dist)

camera = calipy.ColorCamera()
camera.setParamFromMatrix(mtx,dist, 1920, 1080)
camera.saveJson(output_path)
