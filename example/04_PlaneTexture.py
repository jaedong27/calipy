import numpy as np
import cv2
import glob
import sys
sys.path.append("../")
import calipy

cam = calipy.ColorCamera("./CameraData/back/cam_intrinsic.json")
uv_0 = np.dot( cam.intrinsic_inv, np.transpose([[0,0,1]]) )
uv_1 = np.dot( cam.intrinsic_inv, np.transpose([[1920,1080,1]]) )

uv_0 = 10 * uv_0 / uv_0[2]
uv_1 = 10 * uv_1 / uv_1[2]

#print(uv_0, uv_1)

ren = calipy.vtkRenderer()
ren.addCamera("back_cam", cs=0.5)

TV_width = 1.70
TV_height = 0.95

objectPoints = np.array( [ [0, 0, 0],
                 [TV_width, 0, 0],
                 [0, TV_height,0],
                 [TV_width, TV_height, 0] ] ).astype(np.float64)

#ren.addLines("TV", objectPoints, [0,1,3,2,0])
#print(uv_0)
ren.addPlanWithTexture("TV2", [uv_0[0], uv_0[1], 10], [uv_0[0], uv_1[1], 10], [uv_1[0], uv_0[1], 10], "./display.jpg")
#ren.addPlane("TV2", [0,0,10],np.transpose(uv_1)[0,:],  np.transpose(uv_0)[0,:])
#ren.addPlanWithTexture("TV3", [0,0,0], [-1,1,0], [1,-1,0], "./t.jpg")
ren.setMainCamera()
ren.render()