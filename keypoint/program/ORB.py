import numpy as np
import cv2 as cv

img = cv.imread('box.jpg', cv.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

# store processed img
cv.imwrite('../image_result/ORB.jpg', img)

cv.imshow('ORB', img)
cv.waitKey(0)
cv.destroyAllWindows()
