import numpy as np
import cv2
import glob
import sys
sys.path.append("../")
import calipy

Rt_path = "./CameraData/Rt.json"
TVRt_path = "./Cameradata/TVRt.json"
Rt = calipy.Transform(Rt_path)
TVRt = calipy.Transform(TVRt_path)

#TVRt = TVRt.dot(Rt)

origin = calipy.Transform()
ren = calipy.vtkRenderer()
ren.addCamera("front_cam", cs=0.5)
ren.addCamera("back_cam", Rt.R, Rt.T, cs=0.5)

TV_width = 1.70
TV_height = 0.95

objectPoints = np.array( [ [0,0,0],
                 [TV_width, 0, 0],
                 [0, TV_height,0],
                 [TV_width, TV_height, 0] ] ).astype(np.float64)

tvpoints_on_camera = TVRt.move(np.transpose(objectPoints))
ren.addLines("TV", np.transpose(tvpoints_on_camera), [0,1,3,2,0])
#ren.addCamera("TV_origin", TVRt.R, TVRt.T, cs=0.5)

# Display Input Image
cam = calipy.ColorCamera("./CameraData/back/cam_intrinsic.json")

input_img_path = "./display.jpg"
undistort_img_path = "undistort_display.jpg"

input_img = calipy.lib.imreadKorean(input_img_path)
undistort_img = cam.solveDistortion(input_img)
undistort_img = cv2.flip(undistort_img, 0)
cv2.imwrite(undistort_img_path, undistort_img)

uv_0 = np.dot( cam.intrinsic_inv, np.transpose([[0,0,1]]) )
uv_1 = np.dot( cam.intrinsic_inv, np.transpose([[1920,1080,1]]) )

uv_0 = 10 * uv_0 / uv_0[2]
uv_1 = 10 * uv_1 / uv_1[2]

ren.addPlanWithTexture("TV2", [uv_0[0], uv_0[1], 10], [uv_0[0], uv_1[1], 10], [uv_1[0], uv_0[1], 10], undistort_img_path)
ren.setMainCamera()
ren.render()