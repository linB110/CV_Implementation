import numpy as np
import cv2 as cv

img = cv.imread('box.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# SIFT detector
detector = cv.SIFT_create()
keypoints = detector.detect(gray, None)
img = cv.drawKeypoints(gray, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# store processed img
cv.imwrite('SIFT.jpg', img)

# show result
cv.imshow('SIFT', img)
cv.waitKey(0)
cv.destroyAllWindows()

