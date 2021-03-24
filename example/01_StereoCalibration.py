## Reference Code
## https://github.com/bvnayak/stereo_calibration/blob/master/camera_calibrate.py

import numpy as np
import cv2
import glob
import sys
sys.path.append("../")
import calipy


## Input
front_path = "./CameraData/Front/"
back_path = "./CameraData/back/"

## Output
output_path = "./CameraData/Rt.json"


front_data_list_seed = front_path + "*.png"
back_data_list_seed = back_path + "*.png"

front_intrinsic_path = front_path + "cam_intrinsic.json"
back_intrinsic_path = back_path + "cam_intrinsic.json"

front_cam = calipy.ColorCamera(front_intrinsic_path)
back_cam = calipy.ColorCamera(back_intrinsic_path)

### Init
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((3*4, 3), np.float32)
objp[:, :2] = np.mgrid[0:3, 0:4].T.reshape(-1, 2) * 0.135

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints_front = []  # 2d points in image plane.
imgpoints_back = []  # 2d points in image plane.

### Read Image
front_img_list = glob.glob(front_data_list_seed)
back_img_list = glob.glob(back_data_list_seed)

front_img_list.sort()
back_img_list.sort()

for i, fname in enumerate(front_img_list):
    img_f = cv2.imread(front_img_list[i])
    img_b = cv2.imread(back_img_list[i])

    gray_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_f, corners_f = cv2.findChessboardCorners(gray_f, (3, 4), None)
    ret_b, corners_b = cv2.findChessboardCorners(gray_b, (3, 4), None)

    if ret_f is True and ret_b is True:
        rt = cv2.cornerSubPix(gray_f, corners_f, (11, 11), (-1, -1), criteria)
        rt = cv2.cornerSubPix(gray_b, corners_b, (11, 11), (-1, -1), criteria)

        #print(corners_f.shape)
        #print(corners_f)
        temp = np.flip(corners_f.reshape( 4, 3, 2 ), axis=1)
        corners_f = temp.reshape(12,1,2)
        #print(corners_f)

        # Draw and display the corners
        ret_f = cv2.drawChessboardCorners(img_f, (3, 4), corners_f, ret_f)
        ret_b = cv2.drawChessboardCorners(img_b, (3, 4), corners_b, ret_b)
        
        img_f = cv2.resize(img_f, None, fx=0.4, fy=0.4)
        img_b = cv2.resize(img_b, None, fx=0.4, fy=0.4)

        img_for_check = np.concatenate((img_f, img_b), axis=0)

        #imgpoints_f.append(corners_l)        
        #imgpoints_r.append(corners_r)
        cv2.imshow("img_for_check", img_for_check)
        ch = cv2.waitKey(0)
        if ch == ord('q'):
            exit()
        elif ch == ord('e'):
            break
        elif ch == ord(' '):
            print(fname, " -> Add")
            objpoints.append(objp)
            imgpoints_front.append(corners_f)
            imgpoints_back.append(corners_b)
        else:
            print(fname, " -> Skip!!!")


### Calibration
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_ZERO_TANGENT_DIST
# flags |= cv2.CALIB_RATIONAL_MODEL
# flags |= cv2.CALIB_SAME_FOCAL_LENGTH
# flags |= cv2.CALIB_FIX_K3
# flags |= cv2.CALIB_FIX_K4
# flags |= cv2.CALIB_FIX_K5

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                        cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_front,
    imgpoints_back, front_cam.intrinsic, front_cam.dist, back_cam.intrinsic, 
    back_cam.dist, (1920, 1080),
    criteria=stereocalib_criteria, flags=flags)

print('Intrinsic_mtx_1', M1)
print('dist_1', d1)
print('Intrinsic_mtx_2', M2)
print('dist_2', d2)
print('R', R)
print('T', T)
print('E', E)
print('F', F)

Rt = calipy.Transform()
Rt.setParam(R, T)
Rt.saveJson(output_path)



#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#print(ret)
#print(mtx)
#print(dist)

#camera = calipy.ColorCamera()
#camera.setParamFromMatrix(mtx,dist, 1920, 1080)
#camera.saveJson(output_path)
