import numpy as np
import cv2 as cv

img = cv.imread('box.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

fast = cv.FastFeatureDetector_create()

# find and draw keypoints
kp = fast.detect(img, None)
img = cv.drawKeypoints(img, kp, None, color = (255, 0, 0))

# store processed img
cv.imwrite('../image_result/FAST.jpg', img)

# show result
cv.imshow('FAST', img)
cv.waitKey(0)
cv.destroyAllWindows()

