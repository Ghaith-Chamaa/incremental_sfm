import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def create_bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Creates the sparsity matrix for bundle adjustment.

    :param n_cameras: Number of cameras.
    :param n_points: Number of 3D points.
    :param camera_indices: Array of camera indices for each 2D point observation.
    :param point_indices: Array of 3D point indices for each 2D point observation.
    :return: Sparse matrix representing the Jacobian sparsity pattern.
    """
    m = camera_indices.size * 2
    n = n_cameras * 12 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    row_indices = np.arange(camera_indices.size)
    for s in range(12):
        A[2 * row_indices, camera_indices * 12 + s] = 1
        A[2 * row_indices + 1, camera_indices * 12 + s] = 1

    for s in range(3):
        A[2 * row_indices, n_cameras * 12 + point_indices * 3 + s] = 1
        A[2 * row_indices + 1, n_cameras * 12 + point_indices * 3 + s] = 1

    return A

def project_points(points_3d, camera_params, K):
    """
    Projects 3D points onto the image plane using given camera parameters and intrinsics.

    :param points_3d: (N, 3) array of 3D point coordinates.
    :param camera_params: (M, 12) array of camera parameters (rotation and translation).
    :param K: (3, 3) camera intrinsics matrix.
    :return: List of (num_points_in_camera, 2) projected 2D point coordinates.
    """
    projected_points = []
    for cam_param, point in zip(camera_params, points_3d):
        R = cam_param[:9].reshape(3, 3)
        rvec, _ = cv2.Rodrigues(R)
        t = cam_param[9:].reshape(3, 1)
        point = np.expand_dims(point, axis=0)
        projected, _ = cv2.projectPoints(point, rvec, t, K, distCoeffs=np.array([]))
        projected_points.append(np.squeeze(projected))
    return projected_points

def calculate_reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K):
    """
    Calculates the reprojection error for bundle adjustment.

    :param params: 1D array of all camera parameters and 3D point coordinates.
    :param n_cameras: Number of cameras.
    :param n_points: Number of 3D points.
    :param camera_indices: Array of camera indices for each 2D point observation.
    :param point_indices: Array of 3D point indices for each 2D point observation.
    :param points_2d: (N, 2) array of observed 2D point coordinates.
    :param K: (3, 3) camera intrinsics matrix.
    :return: 1D array of reprojection errors.
    """
    camera_params = params[:n_cameras * 12].reshape((n_cameras, 12))
    points_3d = params[n_cameras * 12:].reshape((n_points, 3))
    projected_points = project_points(points_3d[point_indices], camera_params[camera_indices], K)
    return (np.array(projected_points) - points_2d).ravel()

def do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol):
    """
    Performs bundle adjustment to refine camera poses and 3D point coordinates.

    :param points3d_with_views: List of Point3D_with_views objects.
    :param R_mats: Dictionary mapping resected camera indices to their rotation matrices.
    :param t_vecs: Dictionary mapping resected camera indices to their translation vectors.
    :param resected_imgs: List of indices of resected images.
    :param keypoints: List of lists of cv2.Keypoint objects for each image.
    :param K: (3, 3) camera intrinsics matrix.
    :param ftol: Tolerance for change in the cost function for optimization.
    :return: Tuple containing updated points3d_with_views, R_mats, and t_vecs.
    """
    point_indices_list = []
    points_2d_list = []
    camera_indices_list = []
    points_3d_list = []
    initial_camera_params = []
    camera_index_map = {}
    camera_count = 0

    for img_index in resected_imgs:
        camera_index_map[img_index] = camera_count
        initial_camera_params.append(np.hstack((R_mats[img_index].ravel(), t_vecs[img_index].ravel())))
        camera_count += 1

    for pt3d_idx, pt3d_with_view in enumerate(points3d_with_views):
        points_3d_list.append(pt3d_with_view.point3d.flatten())
        for cam_idx, kpt_idx in pt3d_with_view.source_2dpt_idxs.items():
            if cam_idx not in resected_imgs:
                continue
            point_indices_list.append(pt3d_idx)
            camera_indices_list.append(camera_index_map[cam_idx])
            points_2d_list.append(keypoints[cam_idx][kpt_idx].pt)

    if not points_3d_list:
        print("Warning: No common observations found for bundle adjustment.")
        return points3d_with_views, R_mats, t_vecs

    point_indices = np.array(point_indices_list)
    points_2d = np.array(points_2d_list)
    camera_indices = np.array(camera_indices_list)
    initial_points_3d = np.array(points_3d_list)
    initial_camera_params = np.array(initial_camera_params)

    n_cameras = initial_camera_params.shape[0]
    n_points = initial_points_3d.shape[0]
    initial_params = np.hstack((initial_camera_params.ravel(), initial_points_3d.ravel()))
    sparsity_matrix = create_bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    optimization_result = least_squares(
        calculate_reprojection_error,
        initial_params,
        jac_sparsity=sparsity_matrix,
        verbose=2,
        x_scale='jac',
        loss='linear',
        ftol=ftol,
        xtol=1e-12,
        method='trf',
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K)
    )

    adjusted_camera_params = optimization_result.x[:n_cameras * 12].reshape(n_cameras, 12)
    adjusted_points_3d = optimization_result.x[n_cameras * 12:].reshape(n_points, 3)
    updated_R_mats = {}
    updated_t_vecs = {}

    for true_index, normalized_index in camera_index_map.items():
        updated_R_mats[true_index] = adjusted_camera_params[normalized_index][:9].reshape(3, 3)
        updated_t_vecs[true_index] = adjusted_camera_params[normalized_index][9:].reshape(3, 1)

    for i, pt3d_with_view in enumerate(points3d_with_views):
        pt3d_with_view.point3d = adjusted_points_3d[i].reshape(1, 3)

    return points3d_with_views, updated_R_mats, updated_t_vecs