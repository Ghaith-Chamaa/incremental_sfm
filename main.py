import cv2
import pyntcloud
import numpy as np

import pandas as pd
import bundle_adjustment as b
import matching as m
import reconstruction as r

import matplotlib as mpl
import open3d as o3d

mpl.rcParams["figure.dpi"] = 200


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


def visualize_point_cloud(
    points_3D: np.ndarray, colors: np.ndarray, disparity_map: np.ndarray
) -> None:
    """Visualizes the 3D point cloud using Open3D.

    This function takes 3D points, colors, and a disparity map, filters out
    invalid points (NaN, infinite, or where disparity is zero or negative),
    and visualizes the point cloud using Open3D.

    Args:
        points_3D: 3D point cloud (NumPy array of shape (H, W, 3)).
        colors: Color information for each point (NumPy array of shape (H, W, 3) or (N, 3)).
        disparity_map: Disparity map (NumPy array of shape (H, W)).

    Returns:
    None. The function displays the 3D point cloud in a new window using Open3D.

    Raises:
    Exception: If an error occurs during the visualization process.

    Note:
    - The function does not return any value. It only displays the 3D point cloud in a new window.
    - The function filters out invalid points before visualizing the point cloud.
    - The function normalizes the colors before visualizing the point cloud.
    """
    try:
        # Mask invalid points (NaN, infinite, or where disparity is <= 0)
        mask = (
            ~np.isnan(points_3D[:, :, 0])
            & ~np.isinf(points_3D[:, :, 0])
            & (disparity_map > 0)
        )

        if not np.any(mask):  # Check if there are any valid points before proceeding
            print("Warning: No valid points to visualize after masking.")
            return  # Exit early if there are no valid points

        valid_points = points_3D[mask]
        valid_colors = colors[mask]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(
            valid_points
        )  # No need to reshape now
        point_cloud.colors = o3d.utility.Vector3dVector(
            valid_colors / 255.0
        )  # Normalize colors

        o3d.visualization.draw_geometries([point_cloud])

    except Exception as e:
        print(f"Error in visualize_point_cloud: {e}")


def main(n_imgs_, imgset_):
    """
    Main function to perform Structure from Motion (SfM) reconstruction.

    Args:
        n_imgs_ (int): Number of images in the dataset.
        imgset_ (str): Name or identifier of the image set.

    This function performs the following steps:
    1. Extracts features from the images.
    2. Matches features between images.
    3. Removes outliers from the matches.
    4. Initializes the 3D reconstruction.
    5. Iteratively adds new images to the reconstruction.
    6. Performs bundle adjustment to refine the reconstruction.
    7. Visualizes the final 3D point cloud.
    """
    n_imgs = n_imgs_

    # Step 1: Extract features (keypoints and descriptors) from the images
    images, keypoints, descriptors, K = m.find_features(n_imgs, imgset=imgset_)

    # Step 2: Match features between images using a brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_L1)
    matches = m.find_matches(matcher, keypoints, descriptors)
    print("num_matches before outlier removal:", m.num_matches(matches))
    m.print_num_img_pairs(matches)

    # Step 3: Remove outliers from the matches
    matches = m.remove_outliers(matches, keypoints)
    print("After outlier removal:")
    m.print_num_img_pairs(matches)

    # Step 4: Create an adjacency matrix to represent image pairs
    img_adjacency, list_of_img_pairs = m.create_img_adjacency_matrix(n_imgs, matches)

    # Step 5: Initialize the 3D reconstruction using the best image pair
    best_pair = r.best_img_pair(img_adjacency, matches, keypoints, K, top_x_perc=0.2)
    R0, t0, R1, t1, points3d_with_views = r.initialize_reconstruction(
        keypoints, matches, K, best_pair[0], best_pair[1]
    )

    # Store the rotation and translation matrices for the initial pair
    R_mats = {best_pair[0]: R0, best_pair[1]: R1}
    t_vecs = {best_pair[0]: t0, best_pair[1]: t1}

    # List of resected (reconstructed) and unresected (not yet reconstructed) images
    resected_imgs = [best_pair[0], best_pair[1]]
    unresected_imgs = [i for i in range(len(images)) if i not in resected_imgs]
    print("initial image pair:", resected_imgs)
    avg_err = 0

    # Checkpoints for bundle adjustment
    BA_chkpts = [3, 4, 5, 6] + [int(6 * (1.34**i)) for i in range(25)]

    # Step 6: Iteratively add new images to the reconstruction
    while len(unresected_imgs) > 0:
        # Select the next image pair to add to the reconstruction
        resected_idx, unresected_idx, prepend = r.next_img_pair_to_grow_reconstruction(
            n_imgs, best_pair, resected_imgs, unresected_imgs, img_adjacency
        )

        # Get 3D-2D correspondences for Perspective-n-Point (PnP) algorithm
        points3d_with_views, pts3d_for_pnp, pts2d_for_pnp, triangulation_status = (
            r.get_correspondences_for_pnp(
                resected_idx, unresected_idx, points3d_with_views, matches, keypoints
            )
        )

        # Skip if there are too few correspondences for PnP
        if len(pts3d_for_pnp) < 12:
            print(
                f"{len(pts3d_for_pnp)} is too few correspondences for pnp. Skipping imgs resected:{resected_idx} and unresected:{unresected_idx}"
            )
            print(
                f"Currently resected imgs: {resected_imgs}, unresected: {unresected_imgs}"
            )
            continue

        # Perform PnP to estimate the new camera pose
        R_res = R_mats[resected_idx]
        t_res = t_vecs[resected_idx]
        print(f"Unresected image: {unresected_idx}, resected: {resected_idx}")
        R_new, t_new = r.do_pnp(pts3d_for_pnp, pts2d_for_pnp, K)

        # Update the rotation and translation matrices
        R_mats[unresected_idx] = R_new
        t_vecs[unresected_idx] = t_new

        # Add the new image to the list of resected images
        if prepend == True:
            resected_imgs.insert(0, unresected_idx)
        else:
            resected_imgs.append(unresected_idx)
        unresected_imgs.remove(unresected_idx)

        # Test the reprojection error for the new image
        pnp_errors, projpts, avg_err, perc_inliers = r.test_reproj_pnp_points(
            pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K
        )
        print(
            f"Average error of reprojecting points used to resect image {unresected_idx} back onto it is: {avg_err}"
        )
        print(
            f"Fraction of Pnp inliers: {perc_inliers} num pts used in Pnp: {len(pnp_errors)}"
        )

        # Triangulate new 3D points and reproject them to check errors
        if resected_idx < unresected_idx:
            kpts1, kpts2, kpts1_idxs, kpts2_idxs = r.get_aligned_kpts(
                resected_idx,
                unresected_idx,
                keypoints,
                matches,
                mask=triangulation_status,
            )
            if (
                np.sum(triangulation_status) > 0
            ):  # at least 1 point needs to be triangulated
                points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = (
                    r.triangulate_points_and_reproject(
                        R_res,
                        t_res,
                        R_new,
                        t_new,
                        K,
                        points3d_with_views,
                        resected_idx,
                        unresected_idx,
                        kpts1,
                        kpts2,
                        kpts1_idxs,
                        kpts2_idxs,
                        reproject=True,
                    )
                )
        else:
            kpts1, kpts2, kpts1_idxs, kpts2_idxs = r.get_aligned_kpts(
                unresected_idx,
                resected_idx,
                keypoints,
                matches,
                mask=triangulation_status,
            )
            if (
                np.sum(triangulation_status) > 0
            ):  # at least 1 point needs to be triangulated
                points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = (
                    r.triangulate_points_and_reproject(
                        R_new,
                        t_new,
                        R_res,
                        t_res,
                        K,
                        points3d_with_views,
                        unresected_idx,
                        resected_idx,
                        kpts1,
                        kpts2,
                        kpts1_idxs,
                        kpts2_idxs,
                        reproject=True,
                    )
                )

        # Perform bundle adjustment if necessary
        if (
            0.8 < perc_inliers < 0.95
            or 5 < avg_tri_err_l < 10
            or 5 < avg_tri_err_r < 10
        ):
            # If % of inliers from Pnp is too low or triangulation error is too high, bundle adjust
            points3d_with_views, R_mats, t_vecs = b.do_BA(
                points3d_with_views,
                R_mats,
                t_vecs,
                resected_imgs,
                keypoints,
                K,
                ftol=1e0,
            )

        if (
            len(resected_imgs) in BA_chkpts
            or len(unresected_imgs) == 0
            or perc_inliers <= 0.8
            or avg_tri_err_l >= 10
            or avg_tri_err_r >= 10
        ):
            # If % of inliers from Pnp is very low or triangulation error is very high, bundle adjust with stricter tolerance
            points3d_with_views, R_mats, t_vecs = b.do_BA(
                points3d_with_views,
                R_mats,
                t_vecs,
                resected_imgs,
                keypoints,
                K,
                ftol=1e-1,
            )

        # Calculate and print the average reprojection error across all resected images
        av = 0
        for im in resected_imgs:
            p3d, p2d, avg_error, errors = r.get_reproj_errors(
                im,
                points3d_with_views,
                R_mats[im],
                t_vecs[im],
                K,
                keypoints,
                distCoeffs=np.array([]),
            )
            print(f"Average reprojection error on image {im} is {avg_error} pixels")
            av += avg_error
        av = av / len(resected_imgs)
        print(
            f"Average reprojection error across all {len(resected_imgs)} resected images is {av} pixels"
        )

    # Step 7: Visualize the final 3D point cloud
    num_voxels = 100  # Set to 100 for faster visualization, 200 for higher resolution.
    x, y, z = [], [], []
    for pt3 in points3d_with_views:
        if (
            abs(pt3.point3d[0][0]) + abs(pt3.point3d[0][1]) + abs(pt3.point3d[0][2])
            < 100
        ):
            x.append(pt3.point3d[0][0])
            y.append(pt3.point3d[0][1])
            z.append(pt3.point3d[0][2])
    vpoints = list(zip(x, y, z))
    vpoints = np.array(vpoints)
    np.savez("vpoints_np_array_templeRing", vpoints)
    vpoints_df = pd.DataFrame(
        data=vpoints,
        index=[f"{i}" for i in range(vpoints.shape[0])],
        columns=["x", "y", "z"],
    )
    np.savez("vpoints_df_templeRing", vpoints_df)
    np.savez("points3d_with_views_points_3d0", points3d_with_views[0].point3d)
    np.savez(
        "points3d_with_views_source_2dpt_idxs0", points3d_with_views[0].source_2dpt_idxs
    )
    np.savez("R_mats", R_mats)
    np.savez("t_vecs", t_vecs)

    # Create and visualize the voxel grid
    cloud = pyntcloud.PyntCloud(vpoints_df)
    cloud.add_structure("voxelgrid", n_x=num_voxels, n_y=num_voxels, n_z=num_voxels)
    cloud.structures[
        f"V([{num_voxels}, {num_voxels}, {num_voxels}],[None, None, None],True)"
    ].plot(d=3)

    # Visualize the 3D point cloud using Open3D
    visualize_sfm_open3d(vpoints)


if __name__ == "__main__":

    imgset = "templering"
    n_imgs = 46  # 46 if imgset = 'templering', 49 if imgset = 'Viking'
    profile_code = True

    if profile_code:

        # Installing the modules for profiling: !pip install snakeviz
        import cProfile, pstats

        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        prof_stats_file = "profiler-stats"
        stats.dump_stats(prof_stats_file)
        stats.print_stats()

        ## you can visualize the stats with snakeviz using the follwing termianl command:
        # snakeviz prof_stats_file

    else:
        main()
