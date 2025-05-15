import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from typing import Tuple, List
import os
import glob
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict, Optional, Union

def visualize_point_cloud(points_3D):
    """
    Visualize the 3D point cloud using Open3D.

    Parameters:
    - points_3D: 3D points to visualize.
    """
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points
    pcd.points = o3d.utility.Vector3dVector(
        points_3D.T
    )  # Transpose to match Open3D format

    # Optionally, set colors (here we set all points to red)
    colors = np.array([[1, 0, 0] for _ in range(points_3D.shape[1])])  # Red color
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries(
        [pcd], window_name="3D Point Cloud", width=800, height=600
    )


def detect_features_sift(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detects keypoints and computes descriptors using SIFT (Scale-Invariant Feature Transform).

    This function uses the SIFT algorithm to detect keypoints in an image and compute
    their corresponding descriptors.  SIFT is a robust feature detection algorithm that
    is invariant to scale, rotation, and some changes in viewpoint.

    Args:
        image: The input image (NumPy array, typically grayscale or color).

    Returns:
        A tuple containing:
            - keypoints: A list of cv2.KeyPoint objects, where each KeyPoint represents
                         a detected keypoint (location, size, orientation, etc.).
            - descriptors: A NumPy array of shape (N, 128) containing the SIFT descriptors
                           for the detected keypoints. N is the number of keypoints.
                           Each row is a 128-dimensional vector representing the descriptor.
    """
    sift = cv2.SIFT_create()  # Create a SIFT object.
    keypoints, descriptors = sift.detectAndCompute(image, None)  # Detect and compute.
    print(
        f"SIFT: {len(keypoints)} keypoints detected."
    )  # Print the number of keypoints.
    return keypoints, descriptors  # Return the keypoints and descriptors.


def detect_features_orb(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detects keypoints and computes descriptors using ORB (Oriented FAST and Rotated BRIEF).

    This function uses the ORB (Oriented FAST and Rotated BRIEF) algorithm to detect
    keypoints in an image and compute their corresponding descriptors. ORB is a fast and
    efficient feature detection and description algorithm that is a good alternative
    to SIFT and SURF (which are patented).

    Args:
        image: The input image (NumPy array, typically grayscale or color).

    Returns:
        A tuple containing:
            - keypoints: A list of cv2.KeyPoint objects, where each KeyPoint represents
                         a detected keypoint (location, size, orientation, etc.).
            - descriptors: A NumPy array of shape (N, 32) containing the ORB descriptors
                           for the detected keypoints. N is the number of keypoints.
                           Each row is a 32-byte (256-bit) vector representing the descriptor.
    """
    orb = cv2.ORB_create()  # Create an ORB object.
    keypoints, descriptors = orb.detectAndCompute(image, None)  # Detect and compute.
    print(
        f"ORB: {len(keypoints)} keypoints detected."
    )  # Print the number of keypoints.
    return keypoints, descriptors  # Return the keypoints and descriptors.


def match_features_bf(
    desc1: np.ndarray, desc2: np.ndarray, method: str = "sift"
) -> Tuple[List[List[cv2.DMatch]], List[cv2.DMatch]]:
    """
    Matches feature descriptors using Brute-Force Matcher with optional ratio test.

    This function uses the Brute-Force Matcher to find matching feature descriptors
    between two sets of descriptors (e.g., from two images). It also applies a ratio
    test to filter out bad matches.

    Args:
        desc1: NumPy array of shape (N1, D) containing the descriptors from the first image.
               N1 is the number of keypoints in the first image, and D is the descriptor
               dimensionality (128 for SIFT, 32 for ORB).
        desc2: NumPy array of shape (N2, D) containing the descriptors from the second image.
               N2 is the number of keypoints in the second image, and D is the descriptor
               dimensionality (must be the same as desc1).
        method: String specifying the matching method.  "sift" for SIFT descriptors
                (using L2 norm) or "orb" for ORB descriptors (using Hamming distance).

    Returns:
        A tuple containing:
            - matches: A list of lists of cv2.DMatch objects.  Each inner list contains
                       the k-nearest matches (k=2 in this case) for a keypoint in the
                       first image.
            - good_matches: A list of cv2.DMatch objects representing the good matches
                            after the ratio test has been applied.
    """
    if (
        desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0
    ):  # Check for empty descriptors as well.
        print("Descriptors missing or empty, skipping matching.")
        return [], []

    if method == "sift":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # L2 norm for SIFT.
    elif method == "orb":  # Explicitly check for "orb"
        bf = cv2.BFMatcher(
            cv2.NORM_HAMMING, crossCheck=False
        )  # Hamming distance for ORB.
    else:  # Handle invalid method input
        raise ValueError("Invalid matching method. Choose 'sift' or 'orb'.")

    matches = bf.knnMatch(
        desc1, desc2, k=2
    )  # Find the 2 nearest matches for each descriptor.

    # Ratio test (David Lowe's paper) to remove bad matches.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # If the distance to the best match is
            good_matches.append(m)  # significantly smaller than the distance
            # to the second best match, it's a good match.

    print(
        f"{method.upper()} BF Matcher: {len(matches)} matches found, {len(good_matches)} after ratio test."
    )
    return matches, good_matches


def match_features_flann(
    desc1: np.ndarray, desc2: np.ndarray, method: str = "sift"
) -> Tuple[List[List[cv2.DMatch]], List[cv2.DMatch]]:
    """
    Matches feature descriptors using FLANN-based matcher with optional ratio test.

    This function uses the FLANN (Fast Library for Approximate Nearest Neighbors)
    matcher to find matching feature descriptors between two sets of descriptors
    (e.g., from two images). It also applies a ratio test to filter out bad matches.
    FLANN is often faster than brute-force matching, especially for large numbers
    of descriptors.

    Args:
        desc1: NumPy array of shape (N1, D) containing the descriptors from the first image.
               N1 is the number of keypoints in the first image, and D is the descriptor
               dimensionality (128 for SIFT, 32 for ORB).
        desc2: NumPy array of shape (N2, D) containing the descriptors from the second image.
               N2 is the number of keypoints in the second image, and D is the descriptor
               dimensionality (must be the same as desc1).
        method: String specifying the matching method.  "sift" for SIFT descriptors
                or "orb" for ORB descriptors.

    Returns:
        A tuple containing:
            - matches: A list of lists of cv2.DMatch objects.  Each inner list contains
                       the k-nearest matches (k=2 in this case) for a keypoint in the
                       first image.
            - good_matches: A list of cv2.DMatch objects representing the good matches
                            after the ratio test has been applied.
    """
    if (
        desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0
    ):  # Check for empty descriptors as well.
        print("Descriptors missing or empty, skipping matching.")
        return [], []

    if method == "sift":
        index_params = dict(algorithm=1, trees=5)  # Use KD-Tree for SIFT.
    elif method == "orb":
        index_params = dict(
            algorithm=6, table_number=6, key_size=12, multi_probe_level=2
        )  # Use LSH for ORB.
    else:
        raise ValueError("Invalid matching method. Choose 'sift' or 'orb'.")

    search_params = dict(
        checks=50
    )  # Number of times the trees are searched. Higher values give better accuracy but take more time.

    flann = cv2.FlannBasedMatcher(
        index_params, search_params
    )  # Create the FLANN matcher.

    matches = flann.knnMatch(
        desc1, desc2, k=2
    )  # Find the 2 nearest matches for each descriptor.

    # Ratio test (David Lowe's paper) to remove bad matches.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # If the distance to the best match is
            good_matches.append(m)  # significantly smaller than the distance
            # to the second best match, it's a good match.

    print(
        f"FLANN {method.upper()} Matcher: {len(matches)} matches found, {len(good_matches)} after ratio test."
    )
    return matches, good_matches


def plot_keypoints(image, keypoints, title="Detected Keypoints"):
    image_with_kp = cv2.drawKeypoints(
        image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(image_with_kp, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_matches(img1, kp1, img2, kp2, matches, good_matches, title="Feature Matches"):
    img_matches = cv2.drawMatchesKnn(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    img_good_matches = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_matches)
    plt.title(f"{title} (Before Ratio Test)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_good_matches)
    plt.title(f"{title} (After Ratio Test)")
    plt.axis("off")

    plt.show()


# 2. Undistort Images
def undistort_images(imgL, imgR, K, dist):
    """Undistorts the left and right images."""
    h, w = imgL.shape[:2]
    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        K, dist, None, None, (w, h), cv2.CV_32FC1
    )
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        K, dist, None, None, (w, h), cv2.CV_32FC1
    )

    left_undistorted = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
    right_undistorted = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)
    return left_undistorted, right_undistorted


def get_images(base_path, dataset_path, img_format, use_n_imgs=-1, type_="color"):
    images_paths = sorted(
        glob.glob(
            os.path.join(base_path, "datasets", dataset_path) + "/*." + img_format,
            recursive=True,
        )
    )
    images = []
    if not use_n_imgs==-1 and use_n_imgs<=len(images_paths):
        images_paths=images_paths[:use_n_imgs]
        
    for img_path in images_paths:
        if type_ == "color":
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        elif type_ == "gray":
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("Invalid image type. Choose 'color' or 'gray'.")
        images.append(image)
    return images

def extract_name_K_R_T(image_parameters=""):
    if image_parameters == "":
        print("Image parameters file not found.")
        return None
    cameraMatrix = np.array(
        [
            [
                float(image_parameters[1]),
                float(image_parameters[2]),
                float(image_parameters[3]),
            ],
            [
                float(image_parameters[4]),
                float(image_parameters[5]),
                float(image_parameters[6]),
            ],
            [
                float(image_parameters[7]),
                float(image_parameters[8]),
                float(image_parameters[9]),
            ],
        ]
    )

    rotationMatrix = np.array(
        [
            [
                float(image_parameters[10]),
                float(image_parameters[11]),
                float(image_parameters[12]),
            ],
            [
                float(image_parameters[13]),
                float(image_parameters[14]),
                float(image_parameters[15]),
            ],
            [
                float(image_parameters[16]),
                float(image_parameters[17]),
                float(image_parameters[18]),
            ],
        ]
    )

    transMatrix = np.array(
        [
            [float(image_parameters[19])],
            [float(image_parameters[20])],
            [float(image_parameters[21])],
        ]
    )
    return [image_parameters[0], cameraMatrix, rotationMatrix, transMatrix]


def get_img_params(base_path: str, dataset_path: str, img_params: list = []) -> list:
    """
    Reads the intrinsic and extrinsic camera parameters from a file.

    This function reads the intrinsic and extrinsic camera parameters from a file
    and returns them as NumPy arrays. The file should contain the camera matrix,
    distortion coefficients, rotation and translation vectors for each image.

    Parameters:
    - base_path (str): The base path of the dataset.
    - dataset_path (str): The path of the dataset within the base path.
    - img_params (list, optional): A list to store the image parameters. Defaults to an empty list.

    Returns:
    - list: A list containing the image parameters read from the file. Each element in the list is a list of strings,
            representing the parameters for a single image.

    The file format should be as follows:
    - Each line in the file represents the parameters for a single image.
    - The first element in each line is the image name.
    - The remaining elements in each line are the camera matrix, distortion coefficients, rotation and translation vectors.
    """
    try:
        with open(
            os.path.join(base_path, dataset_path, "" + dataset_path + "_par.txt"), "r"
        ) as file:
            for line in file:
                if line.startswith(dataset_path):
                    img_params.append(
                        extract_name_K_R_T(line.split())
                    )  # imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3
            return img_params
    except FileNotFoundError:
        print(f"File '{dataset_path}_par.txt' not found.")
        return []


def get_sift_features(image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detects keypoints and computes descriptors using SIFT (Scale-Invariant Feature Transform).

    This function uses the SIFT algorithm to detect keypoints in an image and compute
    their corresponding descriptors.  SIFT is a robust feature detection algorithm that
    is invariant to scale, rotation, and some changes in viewpoint.

    Args:
        image: The input image (NumPy array, typically grayscale or color).

    Returns:
        A tuple containing:
            - keypoints: A list of cv2.KeyPoint objects, where each KeyPoint represents
                         a detected keypoint (location, size, orientation, etc.).
            - descriptors: A NumPy array of shape (N, 128) containing the SIFT descriptors
                           for the detected keypoints. N is the number of keypoints.
                           Each row is a 128-dimensional vector representing the descriptor.
    """
    sift = cv2.SIFT_create()  # Create a SIFT object.
    keypoints, descriptors = sift.detectAndCompute(image, None)  # Detect and compute.
    return keypoints, descriptors  # Return the keypoints and descriptors.


def build_histogram(descriptor, kmeans):
    labels = kmeans.predict(descriptor)
    histogram, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    return histogram / np.sum(histogram)  # Normalize histogram

class Point3DWithViews:
    """
    Represents a 3D point along with the indices of its corresponding 2D keypoints in multiple images.

    Attributes:
        point3d (np.ndarray): The 3D coordinates of the point (shape: (3,)).
        source_2dpt_idxs (Dict[int, int]): Mapping from image index to keypoint index within that image.
    """
    def __init__(self, point3d: np.ndarray, source_2dpt_idxs: Dict[int, int]):
        self.point3d = point3d
        self.source_2dpt_idxs = source_2dpt_idxs

# Helper function to convert rotation matrix to quaternion (w, x, y, z)
# This is a common implementation. You might need to adjust based on COLMAP's exact convention
# if it differs, but (w, x, y, z) for R_cw (camera from world) is typical.
def rotation_matrix_to_quaternion(R):
    """Converts a rotation matrix to a quaternion (w, x, y, z)."""
    # Ensure the matrix is a NumPy array
    R = np.asarray(R, dtype=float)
    
    # Calculate the trace of the matrix
    trace = np.trace(R)
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
        
    return qw, qx, qy, qz

def export_to_colmap(
    output_path: str,
    K_matrix: np.matrix,
    image_paths: List[str],
    loaded_images: List[np.ndarray], 
    all_keypoints: List[List[cv2.KeyPoint]],
    reconstructed_R_mats: Dict[int, np.ndarray],
    reconstructed_t_vecs: Dict[int, np.ndarray],
    reconstructed_points3d_with_views: List[Point3DWithViews],
    image_height: int,
    image_width: int,
    point_color_strategy: str = "first" # NEW: "first", "average", or "median"
):
    """
    Exports reconstruction data to COLMAP text format, with selectable RGB strategy.
    """
    os.makedirs(output_path, exist_ok=True)
    valid_color_strategies = ["first", "average", "median"]
    if point_color_strategy not in valid_color_strategies:
        raise ValueError(f"Invalid point_color_strategy. Choose from {valid_color_strategies}")

    # --- 1. cameras.txt (remains the same) ---
    with open(os.path.join(output_path, "cameras.txt"), "w") as f_cam:
        f_cam.write("# Camera list with one line of data per camera:\n")
        f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_cam.write(f"# Number of cameras: 1\n")
        
        cam_id = 1 
        model = "PINHOLE"
        width = image_width
        height = image_height
        fx = K_matrix[0, 0]
        fy = K_matrix[1, 1]
        cx = K_matrix[0, 2]
        cy = K_matrix[1, 2]
        f_cam.write(f"{cam_id} {model} {width} {height} {fx} {fy} {cx} {cy}\n")

    colmap_img_id_map = {py_idx: col_idx for col_idx, py_idx in enumerate(sorted(reconstructed_R_mats.keys()), 1)}
    
    # --- 2. points3D.txt (modified for RGB strategy) ---
    points3d_lines_buffer = [] 

    num_valid_3d_points = 0
    track_lengths = []

    for pt3d_py_idx, pt_obj in enumerate(reconstructed_points3d_with_views):
        colmap_point3d_id = pt3d_py_idx + 1
        
        x_3d, y_3d, z_3d = pt_obj.point3d.ravel()
        
        r_val, g_val, b_val = 0, 0, 0 # Default color
        point_colors_bgr_list = [] # Store as list of (B,G,R) tuples

        sorted_observing_py_img_indices = sorted(pt_obj.source_2dpt_idxs.keys())

        for img_py_idx in sorted_observing_py_img_indices:
            if img_py_idx in reconstructed_R_mats:
                kpt_original_idx = pt_obj.source_2dpt_idxs[img_py_idx]
                
                if 0 <= img_py_idx < len(loaded_images) and \
                   0 <= kpt_original_idx < len(all_keypoints[img_py_idx]):
                    
                    kp = all_keypoints[img_py_idx][kpt_original_idx]
                    kp_x, kp_y = kp.pt
                    iy = int(round(kp_y))
                    ix = int(round(kp_x))

                    if 0 <= iy < image_height and 0 <= ix < image_width:
                        bgr_pixel = loaded_images[img_py_idx][iy, ix]
                        point_colors_bgr_list.append(bgr_pixel) # Store as (B, G, R)
        
        if point_colors_bgr_list:
            if point_color_strategy == "first":
                # Use the color from the first valid observation
                b_val, g_val, r_val = point_colors_bgr_list[0] 
            
            elif point_color_strategy == "average":
                # Average colors (convert to float for mean, then back to int)
                avg_b = int(round(np.mean([c[0] for c in point_colors_bgr_list])))
                avg_g = int(round(np.mean([c[1] for c in point_colors_bgr_list])))
                avg_r = int(round(np.mean([c[2] for c in point_colors_bgr_list])))
                b_val, g_val, r_val = avg_b, avg_g, avg_r
            
            elif point_color_strategy == "median":
                # Median colors (more robust to outliers)
                # np.median operates on flattened arrays or per-axis
                # For RGB, it's common to take median per channel
                median_b = int(round(np.median([c[0] for c in point_colors_bgr_list])))
                median_g = int(round(np.median([c[1] for c in point_colors_bgr_list])))
                median_r = int(round(np.median([c[2] for c in point_colors_bgr_list])))
                b_val, g_val, r_val = median_b, median_g, median_r
            
            # Ensure values are within 0-255 (though rounding from mean/median should be okay)
            r_val = np.clip(r_val, 0, 255)
            g_val = np.clip(g_val, 0, 255)
            b_val = np.clip(b_val, 0, 255)
        
        # Note: COLMAP expects R G B order
        final_r, final_g, final_b = r_val, g_val, b_val

        error = 0.0 

        track_str_parts = []
        current_track_length = 0
        for img_py_idx, kpt_original_idx in pt_obj.source_2dpt_idxs.items():
            if img_py_idx in colmap_img_id_map:
                colmap_image_id = colmap_img_id_map[img_py_idx]
                track_str_parts.extend([str(colmap_image_id), str(kpt_original_idx)])
                current_track_length += 1

        if current_track_length >= 2:
            num_valid_3d_points += 1
            track_lengths.append(current_track_length)
            track_str = " ".join(track_str_parts)
            points3d_lines_buffer.append(f"{colmap_point3d_id} {x_3d} {y_3d} {z_3d} {final_r} {final_g} {final_b} {error} {track_str}\n")

    with open(os.path.join(output_path, "points3D.txt"), "w") as f_pts3d:
        f_pts3d.write("# 3D point list with one line of data per point:\n")
        f_pts3d.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        mean_track_len_val = np.mean(track_lengths) if track_lengths else 0
        f_pts3d.write(f"# Number of points: {num_valid_3d_points}, mean track length: {mean_track_len_val}\n")
        for line in points3d_lines_buffer:
            f_pts3d.write(line)

    # --- 3. images.txt ---
    images_lines_buffer = []
    total_observations_for_header = 0

    for py_img_idx in sorted(reconstructed_R_mats.keys()):
        if py_img_idx not in colmap_img_id_map:
            continue

        colmap_image_id = colmap_img_id_map[py_img_idx]
        R = reconstructed_R_mats[py_img_idx]
        t = reconstructed_t_vecs[py_img_idx].ravel()
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
        tx, ty, tz = t[0], t[1], t[2]
        camera_colmap_id = 1
        img_name = os.path.basename(image_paths[py_img_idx])

        images_lines_buffer.append(f"{colmap_image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_colmap_id} {img_name}\n")

        points2d_line_parts = []
        num_observations_in_image = 0
        
        img_kpt_to_pt3d_id = {}
        for pt3d_py_idx, pt_obj in enumerate(reconstructed_points3d_with_views):
            temp_track_len = 0
            for obs_img_idx in pt_obj.source_2dpt_idxs.keys():
                if obs_img_idx in reconstructed_R_mats:
                    temp_track_len +=1
            
            if temp_track_len >= 2:
                colmap_pt3d_id_current = pt3d_py_idx + 1
                if py_img_idx in pt_obj.source_2dpt_idxs:
                    kpt_orig_idx = pt_obj.source_2dpt_idxs[py_img_idx]
                    img_kpt_to_pt3d_id[(py_img_idx, kpt_orig_idx)] = colmap_pt3d_id_current
        
        for kpt_original_idx, kp in enumerate(all_keypoints[py_img_idx]):
            x_2d, y_2d = kp.pt
            observed_colmap_point3d_id = img_kpt_to_pt3d_id.get((py_img_idx, kpt_original_idx), -1)
            points2d_line_parts.extend([str(x_2d), str(y_2d), str(observed_colmap_point3d_id)])
            if observed_colmap_point3d_id != -1:
                num_observations_in_image += 1
        
        total_observations_for_header += num_observations_in_image
        images_lines_buffer.append(" ".join(points2d_line_parts) + "\n")

    with open(os.path.join(output_path, "images.txt"), "w") as f_img:
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        mean_obs_val = total_observations_for_header / len(colmap_img_id_map) if colmap_img_id_map else 0
        f_img.write(f"# Number of images: {len(colmap_img_id_map)}, mean observations per image: {mean_obs_val}\n")
        for line in images_lines_buffer:
            f_img.write(line)

    print(f"COLMAP data exported to {output_path} using '{point_color_strategy}' color strategy.")


def visualize_sfm_open3d(points_3d):
    """
    Visualizes a 3D point cloud obtained from Structure from Motion (SfM) using Open3D.

    Parameters:
    points_3d (numpy.ndarray): A 2D NumPy array representing the 3D points. Each row contains the (x, y, z) coordinates of a point.

    Returns:
    None. The function displays the 3D point cloud in a new window using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    o3d.visualization.draw_geometries([pcd], window_name="SfM 3D Reconstruction")