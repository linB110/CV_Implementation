import numpy as np
import cv2 as cv

img = cv.imread('box.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# detect corners
corners = cv.goodFeaturesToTrack(gray, 50, 0.01, 10)  # 50-> upper limit of corners
corners = corners.astype(int)

# plotting corners
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, (0, 0, 255), -1)  # use red dot

# store img and show result
cv.imshow('Tomasi', img)
cv.imwrite('../image_result/Tomasi.jpg', img)

cv.waitKey(0)
cv.destroyAllWindows()

