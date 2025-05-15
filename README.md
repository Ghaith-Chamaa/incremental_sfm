# Incremental Structure from Motion (SfM) Project

This repository implements an **Incremental Structure from Motion (SfM)** pipeline for 3D scene reconstruction from a dataset of 2D images. The pipeline extracts 3D point clouds and camera poses by leveraging feature matching, epipolar geometry, triangulation, and bundle adjustment. This README provides a detailed explanation of the theory, methodology, results, discussion, and instructions for running the code.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Theory](#theory)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Discussion](#discussion)
6. [How to Run the Code](#how-to-run-the-code)
7. [References](#references)

---

## Introduction

Structure from Motion (SfM) is a photogrammetric technique used to reconstruct 3D scenes from a set of 2D images. The incremental SfM approach builds the 3D model progressively by initializing reconstruction with a pair of images and incrementally adding more images and 3D points. This project uses a dataset of **100 images** (configurable) and assumes a known camera calibration matrix to perform feature extraction, matching, triangulation, and bundle adjustment for accurate 3D reconstruction.

---

## Theory

### Structure from Motion Overview
SfM estimates 3D structure and camera motion from a sequence of 2D images by exploiting the geometric relationships between images. The pipeline involves:
- **Feature Extraction and Matching**: Identifying and matching keypoints across images to establish correspondences.
- **Epipolar Geometry**: Using the fundamental matrix to filter outliers and build connectivity between images.
- **Triangulation**: Computing 3D points from 2D correspondences.
- **Perspective-n-Point (PnP)**: Estimating camera poses for new images.
- **Bundle Adjustment (BA)**: Optimizing camera parameters and 3D points to minimize reprojection errors.

### Key Concepts
1. **Lowe's Ratio Test**: For keypoint matching, the ratio of distances between the best and second-best matches is computed. A match is reliable if the ratio is below a threshold (e.g., 0.75), ensuring high confidence in the correspondence.
2. **Fundamental Matrix**: A 3x3 matrix (rank 2) that enforces epipolar constraints to filter outlier matches. It relates corresponding points in two images via the epipolar geometry.
3. **Epipolar Graph**: An adjacency matrix representing image pairs with sufficient good matches, where edges are weighted by the number of matches.
4. **Triangulation**: Computes 3D points from 2D correspondences using camera poses and the calibration matrix.
5. **PnP**: Estimates the extrinsic camera parameters (rotation and translation) of a new image given 2D-3D correspondences and the calibration matrix.
6. **Bundle Adjustment**: A non-linear least squares optimization that minimizes reprojection errors by refining camera poses and 3D point coordinates.

### Mathematical Formulation
The bundle adjustment objective is to minimize the reprojection error:

$$
E(\{R_i, T_i\}_{i=1}^m, \{X_j\}_{j=1}^N) = \sum_{i=1}^m \sum_{j=1}^N \theta_{ij} \left\| \tilde{x}_{ji} - \pi(R_i, T_i, X_j) \right\|^2
$$

Where:

- $(R_i, T_i)$: Rotation and translation of camera $i$.
- $X_j$: 3D point $j$.
- $\tilde{x}_{ji}$: Observed 2D keypoint in image $i$ for point $j$.
- $\pi$: Perspective projection function.
- $\theta_{ij}$: Binary indicator (1 if point $j$ is visible in image $i$, 0 otherwise).

The Jacobian matrix for BA is sparse, with non-zero blocks corresponding to camera parameters $(A_{ijk})$ and 3D points $(B_{ijk})$.

---

## Methodology

The SfM pipeline is divided into three main scripts: **matching**, **reconstruction**, and **bundle adjustment**. Below is a detailed breakdown of each step.

### 1. Dataset and Calibration
- **Dataset**: The pipeline processes a dataset of **100 images**. The number of images can be configured in the script.

- **Calibration Matrix**: The camera intrinsic parameters (focal length, principal point, etc.) are assumed known and provided as a 3x3 matrix $K$. Example:

  $$
  K = \begin{bmatrix}
  f_x & 0 & c_x \\
  0 & f_y & c_y \\
  0 & 0 & 1
  \end{bmatrix}
  $$

### 2. Matching Script
The matching script establishes correspondences between image pairs:
1. **Feature Extraction**: Extract keypoints and descriptors (e.g., SIFT) from all images.
2. **All-to-All Matching**: Perform k-nearest neighbor (k-NN) matching between descriptors of all image pairs. For each pair \(i, j\), compute a list of good matches based on Lowe's ratio test (threshold = 0.75).
3. **Outlier Filtering**: Use the fundamental matrix \(F\) to filter outliers. The fundamental matrix enforces epipolar constraints, ensuring geometric consistency. Matches satisfying $\text{rank}(F) = 2$ are retained.
4. **Epipolar Graph Construction**: Build an adjacency matrix (upper triangular) where vertices are images, and edges represent pairs with sufficient good matches. Edge weights are the number of matches. This is referred to as the epipolar graph [Fusiello, P131].

The output is a 3D matrix ($n \times n \times m$) where:
- $n$: Number of images.
- $m$: Number of good matches for a given image pair.

### 3. Reconstruction Script
The reconstruction script performs incremental 3D reconstruction [Fusiello, P144]:
1. **Initial Pair Selection**: Select the best image pair for initial triangulation based on:
   - **Sufficient Translation**: Ensures triangulation stability [Geiger, P33].
   - **High Number of Matches**: Balances translation and feature correspondence count. Matches are normalized as percentages to avoid hardcoding thresholds.
2. **Initial Triangulation**: Use the selected image pair to compute initial 3D points via triangulation. The function `get_aligned_kpts` ensures aligned keypoint indices for consistent access to matched keypoints across images.
3. **Incremental Reconstruction**:
   - For each unresected image, use resected images (those already used in triangulation) to estimate the camera pose via PnP [Stachniss].
   - PnP requires at least 3 2D-3D correspondences and the calibration matrix to solve for 6 unknowns (3 rotations, 3 translations).
   - Add new 3D points from the unresected image to the point cloud via triangulation.
   - The `Point3DWithViews` class tracks 3D points, their 2D origins, and the image pairs from which they were triangulated.
   - The `get_correspondences_for_pnp` function returns index aligned lists of 3D and 2D points to be used for PnP. For each 3D point check if it is seen by the resected image, if so check if there is a match for it between the resected and unresected image. If so that point will be used in PnP. Also keeps track of matches that do not have associated 3D points, and therefore need to be triangulated.


### 4. Bundle Adjustment Script
Bundle adjustment (BA) refines the reconstruction by minimizing reprojection errors [Cremers P05, Fusiello P167]:
1. **Monitoring**: Track PnP inlier percentage and reprojection error. If either is too low, trigger BA.
2. **Optimization**: Solve the non-linear least squares problem using solvers that leverage the sparse Jacobian matrix. 

The Jacobian consists of:
   - $A_{ijk}$: is the matrix of the partial derivatives of the residual of the point \(j\) in frame \(i\) versus frame orientation \(k\), i.e. Partial derivatives of residuals w.r.t. camera parameters.
   
   $$
   A_{ijk} = \frac{\partial \tilde{\eta}(P_i \mathbf{M}^j)}{\partial \mathbf{g}_k^\top}
   $$

   - $B_{ijk}$: is the matrix of the partial derivatives of the residual of the point \(j\) in the frame \(i\) with respect to the coordinates of the point \(k\), i.e. Partial derivatives w.r.t. 3D point coordinates.

   $$
   B_{ijk} = \frac{\partial \tilde{\eta}(P_i \mathbf{M}^j)}{\partial \tilde{\mathbf{M}}_k^\top}
   $$

3. **Sparsity Exploitation**: It is easy to see that $A_{ijk} = 0$ for all $i \ne k$ and $B_{ijk} = 0$ for all $j \ne k$. Hence, the Jacobian matrix has a sparse block structure. In the implementation, we take advantage of this sparsity by using the camera observation indices to activate only the specific camera that made the corresponding 2D–3D observation reducing computational cost.

The Jacobian matrix has a sparse block structure, where each observation only affects one camera (orientation) and one 3D point (structure):

$$
\begin{pmatrix}
A_{111} &         &         & \big| & B_{111} &         &        &        \\
A_{121} &         &         & \big| &        & B_{122} &        &        \\
\vdots  &         &         & \big| &        &         &        & \vdots \\
A_{1n_1}&         &         & \big| &        &         &        & B_{1nn} \\
        & A_{212} &         & \big| & B_{211} &        &        &        \\
        & A_{222} &         & \big| &        & B_{222} &        &        \\
        & \vdots  &         & \big| &        &         &        & \vdots \\
        & A_{2n_2}&         & \big| &        &         &        & B_{2nn} \\
\cdots  & \cdots  & \ddots  & \big| & \cdots & \cdots  & \ddots & \cdots \\
        &         & A_{m1m} & \big| & B_{m11} &         &        &        \\
        &         & A_{m2m} & \big| &         & B_{m22} &        &        \\
        &         & \vdots  & \big| &         &         & \ddots &        \\
        &         & A_{mn_m m} & \big| &     &         &        & B_{mnn}
\end{pmatrix}
$$
## Results

The pipeline produces:
- **3D Point Cloud**: A sparse set of 3D points representing the scene.
- **Camera Poses**: Extrinsic parameters (rotation and translation) for each image.
- **Reprojection Error**: Typically reduced to below 1 pixel after bundle adjustment.
- **Performance**: The pipeline successfully reconstructs scenes with 46 images in reasonable time (dependent on hardware and dataset complexity).

Sample result:
TODO

---

## Discussion

### Strengths
- **Robust Matching**: Lowe's ratio and fundamental matrix filtering ensure high-quality correspondences.
- **Incremental Approach**: Scales well with the number of images by adding one image at a time.
- **Bundle Adjustment**: Significantly improves accuracy by optimizing all parameters jointly.
- **Flexible Initialization**: The percentage-based pair selection avoids hardcoding thresholds.

### Limitations
- **Computational Cost**: All-to-all matching and BA are computationally expensive for large datasets.
- **Calibration Dependency**: Requires a known calibration matrix, which may not always be available.
- **Initialization Sensitivity**: Poor initial pair selection can lead to unstable triangulation.
- **Outlier Sensitivity**: Despite filtering, some outliers may persist in challenging datasets.

### Future Improvements
- Implement parallel processing for feature matching to reduce runtime.
- Add automatic calibration estimation for datasets without known intrinsics.
- Explore deep learning-based feature matching for better robustness.

---

## How to Run the Code

### Prerequisites
- **Python 3.8**
- **Libraries**:
  - `numpy`: Matrix operations.
  - `opencv-python`: Feature extraction, matching, and fundamental matrix estimation.
  - `scipy`: Sparse matrix handling for BA.
  - `open3d`: Visualization of results.
- Install dependencies:
  ```bash
  pip install numpy opencv-python scipy open3d
  ```

### Directory Structure
```
incremental_sfm_project/
├── datasets/
│   ├── templering/        # Dataset images (e.g., image001.jpg, ...)
├── matching.py            # Feature extraction and matching
├── reconstruction.py      # Incremental SfM
├── bundle_adjustment.py   # Bundle adjustment
├── README.md              # This file
└── main.py                # Main script to run the pipeline
```

### Steps to Run
1. **Prepare the Dataset**:
   - Place your images in `datasets/templering/`.
   - Provide the calibration matrix in main.py.
   - Provide Number of images to process in main.py.
   - Provide  Path to image directory in main.py.
2. **Run the Pipeline**:
   ```bash
   python main.py
   ```

3. **Output**:
   - 3D point cloud saved as `output/point_cloud.ply`.

## References
1. Fusiello, A. *Lecture Notes on Computer Vision: 3D Reconstruction Techniques*. University of Udine, IT.
2. Geiger, A. *Lecture Notes on Computer Vision, Lecture 3 – Structure-from-Motion*. Autonomous Vision Group, University of Tübingen.
3. Stachniss, C. *Lecture Notes on Projective 3-Point (P3P) Algorithm / Spatial Resection*. University of Bonn.
4. Cremers, D. *Lecture Notes on Computer Vision II: Multiple View Geometry*. Technical University of Munich.