import numpy as np
import cv2
import glob
import sys
sys.path.append("../")
import calipy


Rt_path = "./CameraData/Rt.json"
TVRt_path = "./Cameradata/TVRt.json"
Rt_back_to_front = calipy.Transform(Rt_path).inv()
Rt_TV_to_back = calipy.Transform(TVRt_path)

Rt_TV_to_front = Rt_back_to_front.dot(Rt_TV_to_back)

#origin = calipy.Transform()
#ren = calipy.vtkRenderer()
#ren.addCamera("front_cam", Rt_TV_to_front.inv().R, Rt_TV_to_front.inv().T, cs=0.3)
#ren.addCamera("back_cam", Rt_TV_to_back.inv().R, Rt_TV_to_back.inv().T, cs=0.5)

#TV_width = 1.70
#TV_height = 0.95

#objectPoints = np.array( [ [0,0,0],
#                 [TV_width, 0, 0],
#                 [0, TV_height,0],
#                 [TV_width, TV_height, 0] ] ).astype(np.float64)

#tvpoints_on_camera = np.transpose(objectPoints)
#ren.addLines("TV", np.transpose(tvpoints_on_camera), [0,1,3,2,0])
##ren.addCamera("TV_origin", TVRt.R, TVRt.T, cs=0.5)
#ren.render()

#exit()


origin = calipy.Transform()
ren = calipy.vtkRenderer()
ren.addCamera("front_cam", cs=0.5)
ren.addCamera("back_cam", Rt_back_to_front.R, Rt_back_to_front.T, cs=0.5)

TV_width = 1.70
TV_height = 0.95

objectPoints = np.array( [ [0,0,0],
                 [TV_width, 0, 0],
                 [0, TV_height,0],
                 [TV_width, TV_height, 0] ] ).astype(np.float64)

tvpoints_on_camera = Rt_TV_to_front.move(np.transpose(objectPoints))
ren.addLines("TV", np.transpose(tvpoints_on_camera), [0,1,3,2,0])
#ren.addCamera("TV_origin", TVRt.R, TVRt.T, cs=0.5)
ren.render()


Rt_back_to_front.saveJson("./CameraData/Rt_back_to_front.json")
Rt_TV_to_front.saveJson("./CameraData/Rt_TV_to_front.json")