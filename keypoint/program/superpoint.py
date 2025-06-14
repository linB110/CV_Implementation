import numpy as np
import cv2 as cv
import random

np.random.seed(42)
random.seed(42)

# read and transform image to gray
gray = cv.imread('../image_result/box.jpg', cv.IMREAD_GRAYSCALE)
h, w = gray.shape

# construct affine transform
def random_affine():
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    perturb = np.random.normal(0, 80, (3, 2)).astype(np.float32)
    pts2 = pts1 + perturb
    return cv.getAffineTransform(pts1, pts2)

# SIFT detector
detector = cv.SIFT_create()

# result container
images = []
keypoints_all = []
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Red, Green, Blue
affine_matrices = [] 

# detect keypoints
for i in range(3):
    M = random_affine()
    affine_matrices.append(M)
    warped = cv.warpAffine(gray, M, (w, h))
    kp = detector.detect(warped, None)
    keypoints_all.append((kp, colors[i]))

    # store each detection result
    img_kp = cv.drawKeypoints(warped, kp, None, color=colors[i],
                              flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(f'img_{i+1}.jpg', img_kp)

# superpose keypoints to original image
img_combined = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
fimg_combined = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

for (kp_list, color), M in zip(keypoints_all, affine_matrices):
    M_inv = cv.invertAffineTransform(M)
    corrected_kp = []
    for kp in kp_list:
        pt = np.array([kp.pt[0], kp.pt[1], 1.0])  # Homogeneous coord
        x, y = np.dot(M_inv, pt)
        new_kp = cv.KeyPoint(x, y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        corrected_kp.append(new_kp)

    img_combined = cv.drawKeypoints(img_combined, corrected_kp, img_combined,
                                    color=color,
                                    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# store and show result
cv.imwrite('../image_result/combined_keypoints.jpg', img_combined)
cv.imshow('../image_result/Combined Keypoints', img_combined)
cv.waitKey(0)
cv.destroyAllWindows()

