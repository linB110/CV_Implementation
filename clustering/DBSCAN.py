import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DEPTH_THRESHOLD = 10.0  # meters

# read associated RGB + Depth
depth = cv2.imread("depth/1341846313.592088.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
rgb = cv2.imread("rgb/1341846313.592026.png")

h, w = depth.shape
mask = (depth > 0) & (depth < DEPTH_THRESHOLD)
ys, xs = np.where(mask)

print(f"Points used: {len(xs)}")

# TUM freiburg3 intrinsics
fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6

# pixel to point cloud
x3d = (xs - cx) * depth[ys, xs] / fx
y3d = (ys - cy) * depth[ys, xs] / fy
z3d = depth[ys, xs]

points_3d = np.column_stack([x3d, y3d, z3d])

points_scaled = np.column_stack([
    x3d,
    y3d,
    z3d * 1.1   # weighted depth value
])

# DBSCAN clustering
dbscan = DBSCAN(
    eps = 0.03,
    min_samples=50   # minimum samples to form a group
)
labels = dbscan.fit_predict(points_scaled)

unique_labels = np.unique(labels)
print(f"Clusters found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")

# visualization
out = rgb.copy()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for i, lab in enumerate(unique_labels):
    if lab == -1:
        continue  # denoise
    
    mask_lab = labels == lab
    
    if np.sum(mask_lab) < 50:
        continue
    
    color = (colors[i][:3] * 255).astype(np.uint8)
    out[ys[mask_lab], xs[mask_lab]] = color

cv2.imwrite("cluster_3d_dbscan.png", out)
print("Saved: cluster_3d_dbscan.png")
