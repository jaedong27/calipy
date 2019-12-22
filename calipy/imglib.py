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

def crop_img(frame, crop_ratio_coord):
    width = frame.shape[1]
    height = frame.shape[0]
    return frame[0 + (int)(height * crop_ratio_coord[2]) : height - (int)(height * crop_ratio_coord[3]), 0 + (int)(width * crop_ratio_coord[0]):width - (int)(width * crop_ratio_coord[1])]

def uncrop(frame, crop_ratio_coord, plane_points):
    width = frame.shape[1]
    height = frame.shape[0]
    for point in plane_points:
        point[0] += (int)(height * crop_ratio_coord[2])
        point[1] += (int)(width * crop_ratio_coord[0])
    return plane_points

def normalize_image(frame):
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
