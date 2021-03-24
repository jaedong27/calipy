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
ren.render()