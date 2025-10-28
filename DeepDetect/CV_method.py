import os, cv2, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

############################################
# Config you may want to tweak
############################################
DATA_ROOT = "/home/lab605/dataset/hpatches-sequences-release"  # <- set your HPatches path
SEQ = "i_leuven"                                            # e.g., "i_leuven", "v_bark", ...
IMG_I, IMG_J = 1, 6                                          # which pair to evaluate (1..6)

# Detector settings: "DEFAULT" or "LOW" (extremely permissive)
THRESHOLD_SETTINGS = "LOW"

# Safety limits to avoid OpenCV pyramid/resize crashes with extreme KP counts
MAX_KP_PER_IMAGE = 200000  # cap very large keypoint sets
MIN_KP_SIZE = 3.0          # enforce a positive size for detectors that return 0

# Try these descriptor heads to pair with all detectors
DESC_CHOICES = ["SIFT", "ORB", "BRISK"]  # you can trim this list

# Lowe's NNDR and RANSAC params
NNDR = 0.9
RANSAC_REPROJ_TH = 3.0               # pixels
RANSAC_MAX_ITERS = 20000
MIN_MATCHES_FOR_H = 8

# Visualization toggles
SAVE_KEYPOINTS = True
SAVE_MATCHES = True
OUTDIR = Path("hpatches_eval_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

############################################
# Utilities
############################################

def read_hpatches_h(path_dir, i=1, j=6):
    base = os.path.join(path_dir, f"H_{i}_{j}")
    cand = [base, base + ".txt", base + ".csv"]
    for p in cand:
        if os.path.exists(p):
            with open(p, "r") as f:
                txt = f.read().strip().replace(",", " ")
            H = np.fromstring(txt, sep=" ")
            if H.size == 9:
                return H.reshape(3, 3)
    raise FileNotFoundError(f"Cannot find homography file among: {cand}")


def make_detector(name: str, mode: str):
    name = name.upper()
    if name == "SIFT":
        if mode == "LOW":
            return cv2.SIFT_create(nfeatures=0, nOctaveLayers=4, contrastThreshold=1e-9, edgeThreshold=100, sigma=0.7)
        return cv2.SIFT_create()
    if name == "ORB":
        # Cap features to avoid downstream compute() issues on some builds
        if mode == "LOW":
            return cv2.ORB_create(nfeatures=min(200_000, MAX_KP_PER_IMAGE), fastThreshold=1)
        return cv2.ORB_create(nfeatures=50_000)
    if name == "BRISK":
        if mode == "LOW":
            return cv2.BRISK_create(thresh=1, octaves=4)
        return cv2.BRISK_create()
    if name == "FAST":
        if mode == "LOW":
            return cv2.FastFeatureDetector_create(threshold=1, nonmaxSuppression=False)
        return cv2.FastFeatureDetector_create()
    if name == "AGAST":
        if mode == "LOW":
            return cv2.AgastFeatureDetector_create(threshold=1, nonmaxSuppression=False, type=cv2.AGAST_FEATURE_DETECTOR_AGAST_7_12D)
        return cv2.AgastFeatureDetector_create()
    raise ValueError(f"Unknown detector: {name}")


def make_descriptor(name: str):
    name = name.upper()
    if name == "SIFT":
        return cv2.SIFT_create(), "float"
    if name == "ORB":
        return cv2.ORB_create(), "binary"
    if name == "BRISK":
        return cv2.BRISK_create(), "binary"
    raise ValueError(f"Unknown descriptor: {name}")


def draw_and_save_keypoints(img, kps, path):
    vis = cv2.drawKeypoints(img, kps, None, (0,242,255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imwrite(str(path), vis)


def sanitize_keypoints(kps, shape):
    """Ensure keypoints are valid for descriptor.compute() on some OpenCV builds.
    - enforce positive size
    - clip to image bounds
    - cap extremely large sets
    """
    h, w = shape[:2]
    fixed = []
    for kp in kps[:MAX_KP_PER_IMAGE]:
        x, y = kp.pt
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        if kp.size <= 0:
            kp.size = MIN_KP_SIZE
        fixed.append(kp)
    return fixed


def flann_match(desc1, desc2, dtype: str):
    if desc1 is None or desc2 is None:
        return []
    if dtype == "binary":
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=15, multi_probe_level=1)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        d1 = np.asarray(desc1, np.uint8)
        d2 = np.asarray(desc2, np.uint8)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        d1 = np.asarray(desc1, np.float32)
        d2 = np.asarray(desc2, np.float32)
    matches = matcher.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if (n.distance + 1e-12) > 0 and (m.distance / (n.distance + 1e-12)) < NNDR:
            good.append(m)
    return good


def find_h_from_matches(kps1, kps2, good_matches):
    if len(good_matches) < MIN_MATCHES_FOR_H:
        return None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_TH, maxIters=RANSAC_MAX_ITERS, confidence=0.999)
    return H, (mask.ravel().tolist() if mask is not None else None)


def corner_transfer_error(H_est, H_gt, shape):
    """Mean corner transfer error (pixels) between H_est and H_gt for image1 corners."""
    h, w = shape[:2]
    corners = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]], dtype=np.float64).T  # 3x4
    def proj(H, P):
        Q = H @ P
        Q /= Q[2:3,:]
        return Q[:2,:].T  # 4x2
    try:
        p_est = proj(H_est, corners)
        p_gt  = proj(H_gt,  corners)
    except Exception:
        return np.inf
    return float(np.mean(np.linalg.norm(p_est - p_gt, axis=1)))


def warp_visual(img1, img2, H, outpath):
    h2, w2 = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w2, h2))
    overlay = np.hstack([img2, warped])
    cv2.imwrite(str(outpath), overlay)

############################################
# Load data
############################################
seq_dir = Path(DATA_ROOT) / SEQ
img1_path = seq_dir / f"{IMG_I}.ppm"
img2_path = seq_dir / f"{IMG_J}.ppm"
if not img1_path.exists() or not img2_path.exists():
    raise FileNotFoundError(f"Images not found: {img1_path} or {img2_path}")

img1 = cv2.imread(str(img1_path))
img2 = cv2.imread(str(img2_path))
H_gt = read_hpatches_h(str(seq_dir), IMG_I, IMG_J)
print("Ground-truth H_{}_{:d}:\n{}".format(IMG_I, IMG_J, H_gt))

############################################
# Evaluate all detectors × descriptors
############################################
DETECTORS = ["SIFT", "ORB", "BRISK", "FAST", "AGAST"]
results = []

for det_name in DETECTORS:
    detector = make_detector(det_name, THRESHOLD_SETTINGS)
    kps1_raw = detector.detect(img1, None)
    kps2_raw = detector.detect(img2, None)

    # Sanitize and cap KPs to avoid OpenCV resize assertions
    kps1 = sanitize_keypoints(kps1_raw or [], img1.shape)
    kps2 = sanitize_keypoints(kps2_raw or [], img2.shape)

    if (kps1 is None or len(kps1) == 0) or (kps2 is None or len(kps2) == 0):
        print(f"[WARN] {det_name}: no usable keypoints; skipping.")
        continue

    if SAVE_KEYPOINTS:
        draw_and_save_keypoints(img1, kps1, OUTDIR / f"{det_name}_img1_kps.png")
        draw_and_save_keypoints(img2, kps2, OUTDIR / f"{det_name}_img2_kps.png")

    for desc_name in DESC_CHOICES:
        descriptor, dtype = make_descriptor(desc_name)
        try:
            kps1d, des1 = descriptor.compute(img1, kps1)
            kps2d, des2 = descriptor.compute(img2, kps2)
        except cv2.error as e:
            print(f"[WARN] {det_name}+{desc_name}: descriptor.compute() failed: {e}")
            continue

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            print(f"[WARN] {det_name}+{desc_name}: descriptor(s) empty; skipping.")
            continue

        good = flann_match(des1, des2, dtype)
        if len(good) < MIN_MATCHES_FOR_H:
            print(f"[WARN] {det_name}+{desc_name}: only {len(good)} good matches; skipping H.")
            continue

        H_est, mask = find_h_from_matches(kps1d, kps2d, good)
        if H_est is None:
            print(f"[WARN] {det_name}+{desc_name}: findHomography failed.")
            continue

        # Metrics
        inliers = int(np.sum(mask)) if mask is not None else 0
        inlier_ratio = inliers / max(len(good), 1)
        cte = corner_transfer_error(H_est, H_gt, img1.shape)

        # Save matches viz
        if SAVE_MATCHES:
            vis_matches_path = OUTDIR / f"{det_name}_{desc_name}_matches.png"
            draw_params = dict(matchColor=(0,242,255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            vis = cv2.drawMatches(img1, kps1d, img2, kps2d, [m for i,m in enumerate(good) if (mask is None) or mask[i]], None, **draw_params)
            cv2.imwrite(str(vis_matches_path), vis)

        # Save warp overlay
        warp_visual(img1, img2, H_est, OUTDIR / f"{det_name}_{desc_name}_warp.png")

        results.append({
            "detector": det_name,
            "descriptor": desc_name,
            "good_matches": len(good),
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "corner_transfer_error(px)": round(cte, 3),
            "H_est": H_est
        })

# Pretty print results (sorted by CTE asc)
if len(results) == 0:
    print("No homographies were estimated. Check paths, parameters, or lower NNDR / raise RANSAC thresholds.")
else:
    results_sorted = sorted(results, key=lambda r: r["corner_transfer_error(px)"])
    from tabulate import tabulate
    table = [[r['detector'], r['descriptor'], r['good_matches'], r['inliers'], r['inlier_ratio'], r['corner_transfer_error(px)']] for r in results_sorted]
    print(tabulate(table, headers=["Detector","Descriptor","Good","Inliers","InlierRatio","CTE (px)"]))

    # Dump all H to a text file for inspection
    with open(OUTDIR / "estimated_H_list.txt", "w") as f:
        for r in results_sorted:
            f.write(f"# {r['detector']}+{r['descriptor']} | good={r['good_matches']} inliers={r['inliers']} ratio={r['inlier_ratio']} CTE={r['corner_transfer_error(px)']}\n")
            np.savetxt(f, r['H_est'], fmt='%.10f')
            f.write("\n")

    print(f"\nSaved visualizations and results to: {OUTDIR.resolve()}")

