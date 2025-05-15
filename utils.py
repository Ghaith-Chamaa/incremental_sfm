import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from typing import Tuple, List
import os
import glob


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


def get_images(base_path, dataset_path, type_="color"):
    images_path = sorted(glob.glob(dataset_path + "/*.png"))
    images = []
    for filename in [os.path.join(base_path, img_path) for img_path in images_path]:
        if filename.endswith(".png"):
            img_path = os.path.join(dataset_path, filename)
            if type_ == "color":
                image = cv2.imread(img_path)
            elif type_ == "gray":
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                raise ValueError("Invalid image type. Choose 'color' or 'gray'.")
            images.append(image)
    return images


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
