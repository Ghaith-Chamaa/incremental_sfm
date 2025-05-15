import cv2
import numpy as np
from typing import List, Tuple


class SIFTMatcher:
    """
    A utility class for extracting SIFT features from images, matching them,
    removing outliers, and generating adjacency information between images.
    """
    def __init__(self, ratio_threshold: float = 0.75, min_matches: int = 20,
                 ransac_threshold: float = 3.0):
        """
        Initialize the SIFT matcher parameters.

        Args:
            ratio_threshold (float): Lowe's ratio test threshold.
            min_matches (int): Minimum number of inlier matches to consider two images connected.
            ransac_threshold (float): RANSAC reprojection threshold for fundamental matrix estimation.
        """
        # Use the modern SIFT API
        self.sift = cv2.SIFT_create()
        # L2 norm is recommended for SIFT descriptors as per our experimentation
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio_threshold = ratio_threshold
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold

    def extract_features(self, images: List[np.ndarray]) -> Tuple[List, List]:
        """
        Detect SIFT keypoints and compute descriptors for a list of images.

        Args:
            images (List[np.ndarray]): List of grayscale images.

        Returns:
            Tuple[List, List]: (keypoints_list, descriptors_list)
        """
        print(f"\n======== Getting Image Features ========")
        keypoints_list: List = []
        descriptors_list: List = []
        i = 0
        for img in images:
            kp, des = self.sift.detectAndCompute(img, None)
            print(f"Extracted {len(kp)} SIFT features for frame# {i}")
            keypoints_list.append(kp)
            descriptors_list.append(des)
            i+=1
        return keypoints_list, descriptors_list

    def match_all_pairs(self, descriptors: List[np.ndarray]) -> List[List]:
        """
        Perform k-NN matching for all unique image pairs.

        Args:
            descriptors (List[np.ndarray]): List of descriptor arrays per image.

        Returns:
            List[List]: Upper-triangular matrix of raw matches.
        """
        print(f"\n======== Image Feature Matching ========")
        n = len(descriptors)
        matches: List[List] = [[[] for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                raw = self.matcher.knnMatch(descriptors[i], descriptors[j], k=2)
                good = [m[0] for m in raw
                        if len(m) == 2 and m[0].distance < self.ratio_threshold * m[1].distance]
                matches[i][j] = good
                print(f"Image# {i} has {len(good)} matches with Image# {j}")
        return matches

    def filter_outliers(self, matches: List[List], keypoints: List) -> List[List]:
        """
        Remove outlier matches using the fundamental matrix and RANSAC.

        Args:
            matches (List[List]): Raw matches for each image pair.
            keypoints (List): List of keypoints per image.

        Returns:
            List[List]: Matches passing geometric consistency.
        """
        print(f"\n\n\n======== Filtering Bad Matches ========\n")
        filtered: List[List] = [[[] for _ in row] for row in matches]
        for i, row in enumerate(matches):
            for j, pair in enumerate(row):
                if j <= i or len(pair) < self.min_matches:
                    continue

                pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in pair])
                pts2 = np.float32([keypoints[j][m.trainIdx].pt for m in pair])

                F, mask = cv2.findFundamentalMat(
                    pts1, pts2, cv2.FM_RANSAC, self.ransac_threshold)

                # If no model found or degenerate
                if F is None or mask is None or (F.shape == (3, 3) and np.linalg.det(F) > 1e-7):
                    continue

                mask_flat = mask.ravel()
                inliers = [pair[k] for k in range(len(pair)) if mask_flat[k] == 1]
                if len(inliers) >= self.min_matches:
                    filtered[i][j] = inliers
                print(f"Image# {i} has {len(inliers)} good matches with Image# {j}")
        return filtered

    def count_total_matches(self, matches: List[List]) -> int:
        """
        Count total matches across all image pairs.

        Args:
            matches (List[List]): Matches matrix.

        Returns:
            int: Sum of match counts.
        """
        return sum(len(pair) for row in matches for pair in row)

    def connectivity(self, matches: List[List]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Build adjacency matrix and list of connected image pairs based on matches.

        Args:
            matches (List[List]): Filtered matches matrix.

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int]]]: (adjacency matrix, connected pairs)
        """
        print(f"\n======== Getting Image Adjacency Matrix ========\n")
        n = len(matches)
        adjacency = np.zeros((n, n), dtype=int)
        connected_pairs: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if matches[i][j]:
                    adjacency[i, j] = 1
                    connected_pairs.append((i, j))
        return adjacency, connected_pairs
