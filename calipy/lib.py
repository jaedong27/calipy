import os
import json
import cv2
import numpy as np

def imreadKorean(path):
    try:
        stream = open(path.encode("utf-8") , "rb")
    except:
        return None
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

def imwriteKorean(path, img):
    img_data = cv2.imencode(".png", img)[1]
    #stream = open(path.encode("utf-8") , "w")
    #bytes = bytearray(stream.read())
    #numpyArray = np.asarray(bytes, dtype=np.uint8)
    with open(path, "wb") as output:
        output.write(img_data)
    #return cv2.imdecode(numpyArray , cv2.IMREAD_UNCHANGED)

def cropImg(frame, crop_ratio_coord):
    width = frame.shape[1]
    height = frame.shape[0]
    return frame[0 + (int)(height * crop_ratio_coord[2]) : height - (int)(height * crop_ratio_coord[3]), 0 + (int)(width * crop_ratio_coord[0]):width - (int)(width * crop_ratio_coord[1])]

def uncropImg(frame, crop_ratio_coord, plane_points):
    width = frame.shape[1]
    height = frame.shape[0]
    for point in plane_points:
        point[0] += (int)(height * crop_ratio_coord[2])
        point[1] += (int)(width * crop_ratio_coord[0])
    return plane_points

def normalizeImg(frame):
    frame = np.clip(frame, 0, 5000) #normalize to 0-5000mm
    frame = np.uint8(frame / 5000.0 * 255)
    return frame

def drawLines(img, points, color = (255,0,0)):
    for v1, v2 in zip(np.transpose(points[:,:-1]), np.transpose(points[:,1:])):
        print(v1, v2)
        cv2.line( img, (int(v1[0]),int(v1[1])), (int(v2[0]),int(v2[1])), color, 1 )
    return img

def getWarpedImageUsingH(img, h): # H scale is in 0.0 ~ 1.0
    width = img.shape[1]
    height = img.shape[0]
    to_proj_scale = np.array([[width, 0, 0],[0, height, 0], [0, 0, 1]])
    to_norm_scale = np.linalg.inv(to_proj_scale)
    to_proj_scale.astype(np.float32)
    to_norm_scale.astype(np.float32)
    h = np.dot(to_proj_scale, np.dot(h, to_norm_scale))
    return cv2.warpPerspective(img, h, (width, height))

def loadJson(path):
    json_data = {}
    if os.path.isfile(path) == False:
        return json_data

    with open(path) as json_file:
        json_data = json.load(json_file)
    return json_data

def saveJson(path, data_dic):
    with open(path, 'w') as outfile:
        json.dump(data_dic, outfile)

def getIntrinsicDataFromRS(intrinsic_string):
    intrinsic = intrinsic_string.split(',')
    
    width = float(intrinsic[0].split(":")[1])
    height = float(intrinsic[1].split(":")[1])
    cx = float(intrinsic[2].split(":")[1]) # ppx
    cy = float(intrinsic[3].split(":")[1]) # ppy
    fx = float(intrinsic[4].split(":")[1])
    fy = float(intrinsic[5].split(":")[1])

    k1 = float(intrinsic[7].split(":")[1][2:])
    k2 = float(intrinsic[8])
    p1 = float(intrinsic[9])
    p2 = float(intrinsic[10][1:])

    return width, height, fx, fy, cx, cy, k1, k2, p1, p2

def getExtrinsicDataFromRS(extrinsic_string):
    extrinsic_string = extrinsic_string.replace("[", "")
    extrinsic_string = extrinsic_string.replace("]", "")
    extrinsic = extrinsic_string.split('\n')
    rotation_str = extrinsic[0][1:].split(":")[1]
    translation_str = extrinsic[1][1:].split(":")[1]
    rotation = np.array(rotation_str.split(",")).astype(np.float)
    rotation = np.reshape(rotation, (3,3))
    translation = np.array(translation_str.split(",")).astype(np.float)
    translation = np.reshape(translation, (3,1))

    return rotation, translation
    
def drawScreen(img, sc):
    #모든 정보는 0~1로 스케일
    height = img.shape[0]
    width = img.shape[1]
    for v in range(sc.v_num):
        for u in range(sc.u_num-1):
            p1 = (int(sc.uv_on_camera[v,u][0] * width), int(sc.uv_on_camera[v,u][1] * height))
            p2 = (int(sc.uv_on_camera[v,u+1][0] * width), int(sc.uv_on_camera[v,u+1][1] * height))
            cv2.line(img, p1, p2, (255,0,0), 1)

    for u in range(sc.u_num):
        for v in range(sc.v_num-1):
            p1 = (int(sc.uv_on_camera[v,u][0] * width), int(sc.uv_on_camera[v,u][1] * height))
            p2 = (int(sc.uv_on_camera[v+1,u][0] * width), int(sc.uv_on_camera[v+1,u][1] * height))
            cv2.line(img, p1, p2, (255,0,0), 1)

    return img

def drawVideoData(img, millis, frame_idx):
    log_string = ("time:%d, idx:%d" % (millis, frame_idx))
    cv2.putText(img,log_string, (100,100), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0),1,cv2.LINE_AA)
    return img

def drawMode(img, mode):
    cv2.putText(img, mode, (100,300), cv2.FONT_HERSHEY_SIMPLEX,1 ,(255,0,0),1,cv2.LINE_AA)
    return img

def drawCorresMode(raw_img, list):
    img = raw_img.copy()
    for idx, pos in enumerate(list):
        u = int(pos[0] * img.shape[1])
        v = int(pos[1] * img.shape[0])
        cv2.circle(img, (u, v), 1, (100,100,200),1)
        cv2.putText(img, str(idx), (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100,100,200), lineType=cv2.LINE_AA)

    return img

def drawResultImg(img, tag_result):
    result_img = img.copy()
    width = 1#img.shape[1]
    height = 1#img.shape[0]
    #print(tag_result)
    for code in tag_result:
        #Draw Line
        cv2.circle(result_img,(int(code.center[0]),int(code.center[1])),3,(0,0,255))
        #print(code.tag_id)
        #print(code.tag_family)
        p1 = (int(code.corners[0,0]*width),int(code.corners[0,1]*height))
        p2 = (int(code.corners[1,0]*width),int(code.corners[1,1]*height))
        cv2.line(result_img, p1, p2, (255,0,0),1)
        p1 = (int(code.corners[1,0]*width),int(code.corners[1,1]*height))
        p2 = (int(code.corners[2,0]*width),int(code.corners[2,1]*height))
        cv2.line(result_img, p1, p2, (0,255,0),1)
        p1 = (int(code.corners[2,0]*width),int(code.corners[2,1]*height))
        p2 = (int(code.corners[3,0]*width),int(code.corners[3,1]*height))
        cv2.line(result_img, p1, p2, (0,0,255),1)
        p1 = (int(code.corners[0,0]*width),int(code.corners[0,1]*height))
        p2 = (int(code.corners[-1,0]*width),int(code.corners[-1,1]*height))
        cv2.line(result_img, p1, p2, (255,255,0),1)
        cv2.putText(result_img, str(code.tag_id), (int(code.center[0]),int(code.center[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,250),2 )
        
    return result_img

def drawHomography(img, H, thickness=2):
    width = img.shape[1]
    height = img.shape[0]
    uv = np.transpose(np.array([[0,0,1],[1,0,1],[0,1,1],[1,1,1]]));
    target_point = H.dot(uv)
    p1 = (int(width * target_point[0,0] / target_point[2,0]), int(height * target_point[1,0] / target_point[2,0]))
    p2 = (int(width * target_point[0,1] / target_point[2,1]), int(height * target_point[1,1] / target_point[2,1]))
    p3 = (int(width * target_point[0,2] / target_point[2,2]), int(height * target_point[1,2] / target_point[2,2]))
    p4 = (int(width * target_point[0,3] / target_point[2,3]), int(height * target_point[1,3] / target_point[2,3]))
    #print(p1,p2,p3,p4)
    if p1[0] < p2[0]:
        cv2.line(img, p1, p2, (0,0,255),thickness)
    if p2[1] < p4[1]:
        cv2.line(img, p2, p4, (0,0,255),thickness)
    if p3[0] < p4[0]:
        cv2.line(img, p4, p3, (0,0,255),thickness)
    if p1[1] < p3[1]:
        cv2.line(img, p3, p1, (0,0,255),thickness)
    return img

def drawHomographyDense(img, H):
    width = img.shape[1]
    height = img.shape[0]
    grid_row = 11;
    grid_col = 11;
    grid = np.indices((grid_row,grid_col))
    u = np.ravel(grid[1]/(grid[1].shape[1]-1));
    v = np.ravel(grid[0]/(grid[0].shape[0]-1));
    #print(grid.shape[1])
    ones = np.ones((1,len(u)))
    #print(u, ones)
    uv = np.concatenate(([u],[v]),axis=0)
    uv = np.concatenate((uv,ones),axis=0)
    target_point = H.dot(uv)
    temp_target_point = target_point.copy()
    target_point = target_point[0:2,:]/ target_point[2,:]
    target_point = np.transpose(target_point)
    target_point = np.reshape(np.ravel(target_point),(grid_row,grid_col,2)) 
    #print(target_point)
    for row_idx in range(len(target_point)-1):
        for col_idx in range(len(target_point[row_idx,:])-1):
            p1 = target_point[row_idx,col_idx]
            p2 = target_point[row_idx,col_idx+1]
            p3 = target_point[row_idx+1,col_idx]
            p4 = target_point[row_idx+1,col_idx+1]
            if p1[0] < p2[0]:
                cv2.line(img, (int(p1[0]*width),int(p1[1]*height)), (int(p2[0]*width),int(p2[1]*height)), (0,0,100),1)
            if p2[1] < p4[1]:
                cv2.line(img, (int(p2[0]*width),int(p2[1]*height)), (int(p4[0]*width),int(p4[1]*height)), (0,0,100),1)
            if p3[0] < p4[0]:
                cv2.line(img, (int(p4[0]*width),int(p4[1]*height)), (int(p3[0]*width),int(p3[1]*height)), (0,0,100),1)
            if p1[1] < p3[1]:
                cv2.line(img, (int(p3[0]*width),int(p3[1]*height)), (int(p1[0]*width),int(p1[1]*height)), (0,0,100),1)
        #if idx % 11 == 0:
        #print(idx, target_point[0, idx] * width, target_point[1, idx] * height, target_point[2, idx])
        #cv2.circle(img,(int(point[0]*width),int(point[1]*height)),5,(0,0,255))
    #cv2.line(img, p9, p5, (255,255,0),2)
    
    #base = np.zeros((1000,1000,3))
    #for idx, point in enumerate(target_point):
    #    cv2.circle(base,(int(point[0]*width/10 + 500),int(point[1]*height/10 + 500)),1,(0,0,255))
    #    cv2.imshow("ttttt",base)
    #    cv2.waitKey(0)
    #cv2.imshow("ttttt",base)
    #cv2.waitKey(0)
    return img


def getHomographyPoint(H):
    uv = np.transpose(np.array([[0,0,1],[1,0,1],[0,1,1],[1,1,1]]));
    target_point = H.dot(uv)
    p1 = ((target_point[0,0] / target_point[2,0]), (target_point[1,0] / target_point[2,0]))
    p2 = ((target_point[0,1] / target_point[2,1]), (target_point[1,1] / target_point[2,1]))
    p3 = ((target_point[0,2] / target_point[2,2]), (target_point[1,2] / target_point[2,2]))
    p4 = ((target_point[0,3] / target_point[2,3]), (target_point[1,3] / target_point[2,3]))
    return np.array([p1,p2,p3,p4])
#    p1 = (int(code.corners[0,0]*width),int(code.corners[0,1]*height))
#    p2 = (int(code.corners[1,0]*width),int(code.corners[1,1]*height))

#    p1 = (int(code.corners[1,0]*width),int(code.corners[1,1]*height))
#    p2 = (int(code.corners[2,0]*width),int(code.corners[2,1]*height))
#    cv2.line(result_img, p1, p2, (0,255,0),1)
#    p1 = (int(code.corners[2,0]*width),int(code.corners[2,1]*height))
#    p2 = (int(code.corners[3,0]*width),int(code.corners[3,1]*height))
#    cv2.line(result_img, p1, p2, (0,0,255),1)
#    p1 = (int(code.corners[0,0]*width),int(code.corners[0,1]*height))
#    p2 = (int(code.corners[-1,0]*width),int(code.corners[-1,1]*height))
#    cv2.line(result_img, p1, p2, (255,255,0),1)

def drawReprojectionData(img, proj_list, H, cam_list):
    if len(proj_list) == 0:
        #print("Point list is empty")
        return img
    if len(proj_list) != len(cam_list):
        print("Point num is not same")
        return img
    
    ones_mat = np.ones((len(proj_list),1))
    proj_points = np.concatenate((np.array(proj_list), ones_mat), axis=1)
    reprojection_points = H.dot(np.transpose(proj_points))
    reprojection_points = reprojection_points[:2,:] / reprojection_points[-1,:]

    cam_points = np.array(cam_list)
    width = img.shape[1]
    height = img.shape[0]
    for idx, cam_point in enumerate(cam_list):
#        print(proj_point, cam_point)
        p1 = (int(reprojection_points[0,idx]*width),int(reprojection_points[1,idx]*height))
        p2 = (int(cam_point[0]*width),int(cam_point[1]*height))
        cv2.line(img, p1, p2, (200,100,50),1)

    return img
    
def getHomography(target_points):
    uv = np.array([[0,0],[1,0],[0,1],[1,1]])
    H, status = cv2.findHomography(uv, target_points)
    return H

def getPointsOnPlaneUsingUV(uv_points, cam_intrinsic_inv, plane_normal, mean): # (uv_points : (-1,3), plane_normal : (1,3))
    uv_points = np.concatenate((uv_points, np.ones((uv_points.shape[0],1))), axis=1)
    uv_dir_on_rs = np.dot(cam_intrinsic_inv, np.transpose(uv_points))
    u_value = np.dot(plane_normal, np.transpose(np.array(mean))) / np.dot(plane_normal, uv_dir_on_rs)
    object_points = u_value * uv_dir_on_rs
    return object_points