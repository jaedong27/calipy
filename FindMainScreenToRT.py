import sys
import numpy as np
import cv2
import os
import calipy
import math

def getWindowPosition():
    window_width = 1.180
    window_height = 0.750
    window_points = [[-window_width/2, -window_height*math.cos((5.0/180.0) * math.pi), -window_height*math.sin((5.0/180.0) * math.pi)],
                    [ window_width/2, -window_height*math.cos((5.0/180.0) * math.pi), -window_height*math.sin((5.0/180.0) * math.pi)],
                    [-window_width/2, 0, 0],
                    [ window_width/2, 0, 0]]
    return np.array(window_points)

def calibrateWindow(intrinsic, distortion, sc):
    distCoeffs = np.zeros((1,4))
    cam_intrinsic = intrinsic
    objectPoints = np.array(getWindowPosition())
    imagePoints = []
    # target_sc[0,0] = [sc[0,0][0] / resize_value,sc[0,0][0] / resize_value]
    # target_sc[0,-1] = [sc[0,-1][0] / resize_value,sc[0,-1][0] / resize_value]
    # target_sc[-1,0] = [sc[-1,0][0] / resize_value,sc[-1,0][0] / resize_value]
    # target_sc[-1,-1] = [sc[-1,-1][0] / resize_value,sc[-1,-1][0] / resize_value]
    imagePoints.append([sc[0,0][0], sc[0,0][1]])
    imagePoints.append([sc[0,-1][0], sc[0,-1][1]])
    imagePoints.append([sc[-1,0][0], sc[-1,0][1]])
    imagePoints.append([sc[-1,-1][0], sc[-1,-1][1]])
    imagePoints = np.array(imagePoints)
    res, Rvec, T = cv2.solvePnP(objectPoints, imagePoints, cam_intrinsic, distCoeffs)
    R, _ = cv2.Rodrigues(Rvec)

    output = np.dot(cam_intrinsic,(np.dot(R,np.transpose(objectPoints)) + T))
    #print(output)
    output = output / output[2,:]
    #print(output)
    #print(objectPoints)
    #print(imagePoints)
    #print(R, T)

    R = np.linalg.inv(R)
    T = -np.dot(R,T)

    print("Window Calibration Result : ", res, R, T)    
    return R, T, output

def drawScreen(img, sc):
    #모든 정보는 0~1로 스케일
    height = img.shape[0]
    width = img.shape[1]
    for v in range(sc.shape[0]):
        for u in range(sc.shape[1]-1):
            p1 = (int(sc[v,u][0] * width), int(sc[v,u][1] * height))
            p2 = (int(sc[v,u+1][0] * width), int(sc[v,u+1][1] * height))
            cv2.line(img, p1, p2, (255,0,0), 1)

    for u in range(sc.shape[0]):
        for v in range(sc.shape[1]-1):
            p1 = (int(sc[v,u][0] * width), int(sc[v,u][1] * height))
            p2 = (int(sc[v+1,u][0] * width), int(sc[v+1,u][1] * height))
            cv2.line(img, p1, p2, (255,0,0), 1)

    return img

def runScreenUI(camera_path, img_path):
    cam = calipy.ColorCamera(camera_path)

    resize_value = 0.5
    original_img = cv2.imread(img_path)
    img = cam.solveDistortion(original_img)
    img = cv2.resize(img, (int(img.shape[1] * resize_value),int(img.shape[0] * resize_value)))

    sc = np.zeros((2,2,2))
    sc[0,0] = [0.202,0.262]
    sc[0,-1] = [0.816,0.28]
    sc[-1,0] = [0.204,0.783]
    sc[-1,-1] = [0.797,0.795]

    ui_selected_idx = -1

    while True:
        frame = img.copy()
        resize_img = drawScreen(frame, sc)
        cv2.imshow('frame',resize_img)
        chr = cv2.waitKey(0)# & 0xFF

        if chr == ord('1'):
            ui_selected_idx = 0
        if chr == ord('2'):
            ui_selected_idx = 1
        if chr == ord('3'):
            ui_selected_idx = 2
        if chr == ord('4'):
            ui_selected_idx = 3
        if chr == ord('r'):
            print("RRRR")
            
        if chr == ord('q'):
            exit(1)
        if chr == ord('y'):
            break

        if ui_selected_idx != -1:
            print(ui_selected_idx)
            if ui_selected_idx == 0:
                u_idx = 0
                v_idx = 0
            if ui_selected_idx == 1:
                u_idx = -1
                v_idx = 0
            if ui_selected_idx == 2:
                u_idx = 0
                v_idx = -1
            if ui_selected_idx == 3:
                u_idx = -1
                v_idx = -1
            if chr == ord('w'):
                sc[v_idx,u_idx][1] -= 0.001
            if chr == ord('a'):
                sc[v_idx,u_idx][0] -= 0.001
            if chr == ord('s'):
                sc[v_idx,u_idx][1] += 0.001
            if chr == ord('d'):
                sc[v_idx,u_idx][0] += 0.001
            if chr == ord('i'):
                sc[v_idx,u_idx][1] -= 0.1
            if chr == ord('j'):
                sc[v_idx,u_idx][0] -= 0.1
            if chr == ord('k'):
                sc[v_idx,u_idx][1] += 0.1
            if chr == ord('l'):
                sc[v_idx,u_idx][0] += 0.1

    print(sc)
    target_sc = np.zeros((2,2,2))
    target_sc[0,0] = [sc[0,0][0] * original_img.shape[1],sc[0,0][1] * original_img.shape[0]]
    target_sc[0,-1] = [sc[0,-1][0] * original_img.shape[1],sc[0,-1][1] * original_img.shape[0]]
    target_sc[-1,0] = [sc[-1,0][0] * original_img.shape[1],sc[-1,0][1] * original_img.shape[0]]
    target_sc[-1,-1] = [sc[-1,-1][0] * original_img.shape[1],sc[-1,-1][1] * original_img.shape[0]]
    print(target_sc)

    R, T, output = calibrateWindow(cam.intrinsic, cam.distortion, target_sc)
    print("output : ", np.transpose(output))
    print("target_sc : ", target_sc)

    rt = calipy.Transform()
    rt.setParam(R,T)
    rt.saveJson("Gopro_pos.json")
    cv2.destroyAllWindows()

if __name__=="__main__":
    camera_path = "cam_gopro7.json"
    img_path = "Gopro.jpg"
    runScreenUI(camera_path, img_path)
