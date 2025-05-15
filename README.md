# incremental_sfm_vibot

# Incremental Structure from Motion (SfM) for 3D Reconstruction: A Master's Semester Project Guide
---
## **1. Introduction**

Structure from Motion (SfM) reconstructs 3D scenes from 2D image sequences by estimating camera poses and 3D point geometry. This project focuses on **incremental SfM**, where the reconstruction grows progressively by adding images one by one. Students will implement a full pipeline, from image acquisition to 3D reconstruction, while exploring theoretical foundations and research challenges.

---
## **2. Project Overview**

### **Objective**

Develop an incremental SfM pipeline to reconstruct a 3D point cloud of an object using either:

1. **Custom Dataset**: Images captured via smartphone (requires camera calibration).
2. **Existing Dataset**: Pre-calibrated datasets (e.g., Middlebury Temple Ring dataset).

### **Key Components**

1. **Camera Calibration** (*Conditional*): Required only for custom datasets.
2. **Feature Detection & Matching**: Establish correspondences across images.
3. **Initial Reconstruction**: Recover camera poses and triangulate initial 3D points.
4. **Incremental Expansion**: Add new cameras via PnP, triangulate points, and refine via bundle adjustment.
5. **Colorization** (*Optional*): Assign RGB values to 3D points for enhanced visualization.

---
## **3. Theoretical Foundations**

### **3.1 Camera Calibration**
**Theory**:
Camera calibration estimates the **intrinsic matrix** $K$ and **distortion coefficients** to correct lens distortion. The intrinsic matrix is:

$$

K = \begin{bmatrix}

f_x & 0 & c_x \\

0 & f_y & c_y \\

0 & 0 & 1

\end{bmatrix}

$$

**Procedure**:
- **Custom Dataset**:
	1. Capture 8-10 images of a checkerboard.
	2. Use OpenCV’s `cv2.findChessboardCorners` and `cv2.calibrateCamera` to compute $K$.
	
- **Existing Dataset**:
	- Use the provided $K$ and omit distortion correction.
	
**Research Challenges**:
- Handling calibration for datasets with varying intrinsics.

---
### **3.2 Feature Detection & Matching**

**Theory**:
- **Feature Detection**: Identify stable keypoints (e.g., SIFT, ORB).
- **Feature Matching**: Establish correspondences using nearest-neighbor search and outlier rejection.

**Mathematics**:
- **Fundamental Matrix** $F$: For matches $(x_i, x_j)$, $x_j^T F x_i = 0$.
- **Lowe’s Ratio Test**:
$$ \frac{\|d_{query} - d_{train}^{(1)}\|}{\|d_{query} - d_{train}^{(2)}\|} < 0.7 $$

**Implementation**:

1. Detect features using `cv2.SIFT_create()`.
2. Match descriptors with `cv2.BFMatcher`.
3. Filter outliers using RANSAC and `cv2.findFundamentalMat()`.
---
### **3.3 Initial Reconstruction**

**Theory**:
Select an initial image pair with sufficient **baseline** (large relative translation/rotation) to triangulate stable 3D points.
**Mathematics**:
1. Compute the **essential matrix** $E$:
$$ E = K^T F K $$
2. Decompose $E$ into $R$ and $t$ via SVD.

**Implementation**:
1. Validate solutions using **cheirality** (positive depth) via `cv2.recoverPose()`.
2. Triangulate points using `cv2.triangulatePoints()`.
---
### **3.4 Incremental Camera Registration (PnP)**

**Theory**:
For each new image, estimate its pose using 3D-2D correspondences (**Perspective-n-Point**).

**Mathematics**:

$$ \min_{R, t} \sum_i \|x_i - \pi(K(RX_i + t))\|^2 $$
**Implementation**:

1. Use RANSAC with `cv2.solvePnPRansac()`.
2. Convert rotation vector to matrix via `cv2.Rodrigues()`.
---
### **3.5 Triangulation & Bundle Adjustment**

**Theory**:
- **Triangulation**: Solve $x_i = P_i X$, $x_j = P_j X$ via SVD.
- **Bundle Adjustment**: Minimize reprojection error:
$$ \min_{\{R_i, t_i, X_j\}} \sum_{i,j} \|x_{ij} - \pi(K(R_i X_j + t_i))\|^2 $$

  
**Implementation**:
1. Use `cv2.triangulatePoints()` for linear triangulation.
2. Refine with `scipy.optimize.least_squares()` or Ceres/g2o.

  

---

  ### **3.6 Colorization** (*Optional*)

**Theory**:
Assign RGB values to 3D points by projecting them back to source images.

**Mathematics**:
- For a 3D point $X$, project to image $i$:
$$ x_i = \pi(K[R_i | t_i]X) $$
- Average colors from visible views.

**Implementation**:
1. Use `cv2.projectPoints()` to map 3D points to 2D coordinates.
2. Sample RGB values and blend across views.

**Research Challenges**:

- Handling occlusions and lighting variations.
---
## **4. Deliverables**

### **4.1 Code Implementation**

1. **Feature Matching**: Detect SIFT/ORB features and filter outliers.
2. **Initial Reconstruction**: Compute $E$, recover $R/t$, triangulate points.
3. **Incremental SfM**: Add ≥5 new cameras via PnP and triangulate points.
4. **Colorization** (*Optional*): Assign RGB values to 3D points.
5. **Camera Calibration** (*Conditional*): Only for custom datasets.

### **4.2 Report**

1. **Mathematical Formulations**: Document mathematical formulations, implementation choices, and challenges.
2. **Calibration Analysis**:
	- Custom dataset: Explain calibration steps.
3. **Results**: Reprojection errors, point cloud visualizations, and camera trajectories.

### **4.3 Visualization**

1. Generate a **3D point cloud** in PLY/XYZ format.
2. Visualize camera poses and trajectories using Open3D.
3. **Colorized Cloud** (*Optional*): Include RGB values for bonus marks.
---
## **5. Grading Scheme**

**Total Marks: 20**

| **Component**                 | **Criteria**                                                                                  | **Marks**  | **Bonus Marks**               |
|--------------------------------|--------------------------------------------------------------------------------------------|------------|-------------------------------|
| **1. Camera Calibration**      | Custom dataset: Calibrate intrinsics $K$                                                   | -          | **2**                         |
| **2. Feature Detection & Matching** | Detect SIFT/ORB features, match with Lowe’s ratio test, and filter outliers via $F$-matrix. | **2**      | -                             |
| **3. Initial Reconstruction**  | Compute $E$, decompose $R/t$, validate via cheirality, and triangulate initial points.     | **3**      | -                             |
| **4. Incremental Expansion**   | Register ≥5 new cameras via PnP (RANSAC) and triangulate new points.                        | **3**      | -                             |
| **5. Bundle Adjustment (BA)**  | Implement BA to refine poses/points and analyze reprojection error reduction.              | **4**      | **+3** for using Ceres/g2o    |
| **6. Visualization**           | Submit 3D point cloud, camera trajectory visualization.                                    | **3**      | -                             |
| **7. Report and Presentation** | A detailed technical report along with a presentation.                                    | **3 + 2**  | -                             |
| **Bonus Tasks**                | - **Colorization**: Assign RGB values to points.<br>- **Custom Dataset**: 30+ images.<br>- **Loop Closure**: Global refinement. | -          | **+1 per task** (Max +3) |

### **Notes**:

1. **Bonus Marks**: Up to **+3** for advanced tasks.
2. **Penalties**:
	- Incomplete code per section: **-2 marks**.
	- Missing visualization: **-1 mark**.

  

---
## **6. Advanced Topics for Research**

1. **Robust Initialization**: Homography for planar scenes.
2. **Scalable BA**: Keyframe-based optimization.
3. **Deep Learning**: Replace SIFT with SuperPoint features.

---
## **7. Resources**

1. **Datasets**:
	- Middlebury Temple Ring: https://vision.middlebury.edu/mview/data/
2. **Libraries**: OpenCV, SciPy, Ceres Solver, Open3D.
3. **Books**: *Multiple View Geometry in Computer Vision* (Hartley & Zisserman).

---
