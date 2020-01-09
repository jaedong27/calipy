import calipy
import cv2

cam = calipy.ColorCamera("cam_param.json")
img = cv2.imread("1.png")
img = cam.solveDistortion(img)
cv2.imshow("test",img)
cv2.waitKey(0)