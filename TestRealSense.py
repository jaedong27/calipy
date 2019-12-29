import cv2
import calipy
import numpy as np

if __name__=="__main__":
    #rs_cam = RGBDCamera("depth_intrin.json", "color_intrin.json", "depth_to_color_extrin.json")
    #depth_img = calipy.lib.imreadKorean("depth.png")
    #color_img = calipy.lib.imreadKorean("color.png")
    #pointcloud = rs_cam.depth_camera.getPointCloudFromDepthImage(depth_img)
    #tex_img = rs_cam.getPointcloudTexture(pointcloud, color_img)

    #tr = calipy.Transform("./calipy/Data/extrinsic/rs_color_to_depth_capture.json")
    #tr_inv = tr.inv()
    #tr_inv.saveJson("./calipy/Data/extrinsic/rs_depth_to_color_capture.json")

    # tr = calipy.Transform("./calipy/Data/extrinsic/rs_depth_to_color_capture.json")
    # tr2 = tr.dot(tr)
    # tr2.saveJson("./calipy/Data/extrinsic/rs_depth_to_color_capture_2.json")

    rs_cam = calipy.RGBDCamera("./calipy/Data/Intrinsic/rs_depth_intrinsic_capture.json","./calipy/Data/Intrinsic/rs_color_intrinsic_capture.json","./calipy/Data/extrinsic/rs_depth_to_color_capture_2.json")
    print(rs_cam.transform)
    depth_img = calipy.lib.imreadKorean("./calipy/Data/Sample/Calib/depth_101_9_2019_12_04.png")
    color_img = calipy.lib.imreadKorean("./calipy/Data/Sample/Calib/color_101_9_2019_12_04.png")

    pointcloud = rs_cam.depth_camera.getPointCloudFromDepthImage(depth_img)
    tex_img = rs_cam.getPointcloudTexture(pointcloud, color_img)

    print(rs_cam.depth_camera.height, rs_cam.depth_camera.width)
    cv2.imshow("test", np.reshape(np.transpose(tex_img), (rs_cam.depth_camera.height, rs_cam.depth_camera.width, 3)))
    cv2.waitKey(0)

    ren = calipy.vtkRenderer()
    ren.drawPoints("test", np.transpose(pointcloud), np.transpose(tex_img))
    ren.render()