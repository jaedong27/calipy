import numpy as np
import cv2
import sys
sys.path.append("../")
import calipy
import copy

target_img_path = "./Cameradata/display.png"
cam_param_path = "./Cameradata/Back/cam_intrinsic.json"

output_path = "./Cameradata/TVRT.json"

img = calipy.lib.imreadKorean(target_img_path)
cam_param = calipy.ColorCamera(cam_param_path)
img = cam_param.solveDistortion(img)

offset = 300
uv_points = np.array([ [offset,offset], 
                       [img.shape[1] - offset, offset], 
                       [offset, img.shape[0] - offset], 
                       [img.shape[1] - offset, img.shape[0] - offset] ] )

uv_points = np.array([ [656, 351], 
                       [1085, 380], 
                       [645, 594], 
                       [1075, 620] ] )


def drawImage(img, uv_points, current_idx):
    temp_img = copy.deepcopy(img)
    cv2.line( temp_img, (int(uv_points[0, 0]), int(uv_points[0, 1])), (int(uv_points[1, 0]), int(uv_points[1, 1])), (255,0,0) )
    cv2.line( temp_img, (int(uv_points[1, 0]), int(uv_points[1, 1])), (int(uv_points[3, 0]), int(uv_points[3, 1])), (255,0,0) )
    cv2.line( temp_img, (int(uv_points[3, 0]), int(uv_points[3, 1])), (int(uv_points[2, 0]), int(uv_points[2, 1])), (255,0,0) )
    cv2.line( temp_img, (int(uv_points[2, 0]), int(uv_points[2, 1])), (int(uv_points[0, 0]), int(uv_points[0, 1])), (255,0,0) )

    cv2.circle( temp_img, (int(uv_points[current_idx, 0]), int(uv_points[current_idx, 1])), 10, (0,0,255))

    return temp_img
        
current_idx = 0
while True:
    img_with_line = drawImage(img, uv_points, current_idx)
    cv2.imshow("test", img_with_line)

    ch = cv2.waitKey(0)
    if ch == ord("e"):
        break
    elif ch == ord("q"):
        exit()
    elif ch == ord("1"):
        current_idx = 0
    elif ch == ord("2"):
        current_idx = 1
    elif ch == ord("3"):
        current_idx = 2
    elif ch == ord("4"):
        current_idx = 3
    
    elif ch == ord("w"):
        uv_points[current_idx, 1] -= 1
    elif ch == ord("a"):
        uv_points[current_idx, 0] -= 1
    elif ch == ord("s"):
        uv_points[current_idx, 1] += 1
    elif ch == ord("d"):
        uv_points[current_idx, 0] += 1

    elif ch == ord("i"):
        uv_points[current_idx, 1] -= 50
    elif ch == ord("j"):
        uv_points[current_idx, 0] -= 50
    elif ch == ord("k"):
        uv_points[current_idx, 1] += 50
    elif ch == ord("l"):
        uv_points[current_idx, 0] += 50

TV_width = 1.70
TV_height = 0.95

distCoeffs = cam_param.dist
cam_intrinsic = cam_param.intrinsic
print(cam_intrinsic)
objectPoints = np.array( [ [0,0,0],
                 [TV_width, 0, 0],
                 [0, TV_height,0],
                 [TV_width, TV_height, 0] ] ).astype(np.float64)
imagePoints = []
imagePoints.append([uv_points[0,0], uv_points[0,1]])
imagePoints.append([uv_points[1,0], uv_points[1,1]])
imagePoints.append([uv_points[2,0], uv_points[2,1]])
imagePoints.append([uv_points[3,0], uv_points[3,1]])
imagePoints = np.array(imagePoints).astype(np.float64)

print(imagePoints.dtype, objectPoints.dtype)

#res, Rvec, T = cv2.solvePnP(objectPoints, imagePoints, cam_intrinsic, distCoeffs)
res, Rvec, T = cv2.solvePnP(objectPoints, imagePoints, cam_intrinsic, np.array([0,0,0,0,0]))
R, _ = cv2.Rodrigues(Rvec)

output = np.dot(cam_intrinsic,(np.dot(R,np.transpose(objectPoints)) + T))
print(output)
output = output / output[2,:]
print(output)
print(objectPoints)
print(imagePoints)
print(R, T)

#R = np.linalg.inv(R)
#T = -np.dot(R,T)

print("Window Calibration Result : ", res, R, T)    
Rt = calipy.Transform()
Rt.setParam(R, T)
Rt.saveJson(output_path)

