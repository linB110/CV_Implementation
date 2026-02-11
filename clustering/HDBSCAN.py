import cv2
import numpy as np
import hdbscan
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
z3d = depth[ys, xs]
x3d = (xs - cx) * z3d / fx
y3d = (ys - cy) * z3d / fy

points_3d = np.column_stack([x3d, y3d, z3d])

# depth weighting
points_weighted = np.column_stack([
    x3d * 1.0,
    y3d * 1.0,
    z3d * 2.0   
])

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,      # group size
    min_samples=50,            # denoise
    cluster_selection_epsilon=0.02
)

labels = clusterer.fit_predict(points_weighted)

unique_labels = np.unique(labels)
num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

print(f"Clusters found: {num_clusters}")

# visualization
out = rgb.copy()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

for i, lab in enumerate(unique_labels):
    if lab == -1:
        continue  
    
    mask_lab = labels == lab
    
    if np.sum(mask_lab) < 80:
        continue
    
    color = (colors[i][:3] * 255).astype(np.uint8)
    out[ys[mask_lab], xs[mask_lab]] = color

cv2.imwrite("cluster_3d_hdbscan.png", out)
print("Saved: cluster_3d_hdbscan.png")
