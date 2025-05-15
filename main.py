import cv2
import pyntcloud
import random
import os
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import bundle_adjustment as b
from  matching import *
from reconstruction import *
from utils import *

base_path = os.getcwd()
USE_PYTORCH_OPTIMIZER = False
SHOW_PLOTS_INTERACTIVELY = False

n_imgs = 46  # 46 if imgset = 'templering', 49 if imgset = 'Viking'
imgset = "templeRing"
K = np.matrix("1520.40 0.00 302.32; 0.00 1525.90 246.87; 0.00 0.00 1.00")
type_ = "png"

# imgset = "dino"
# n_imgs = 46
# K = np.matrix(
#     "3310.400000 0.000000 316.730000; 0.000000 3325.500000 200.550000; 0.000000 0.000000 1.000000"
# )
# type_ = "png"

# n_imgs = 18
# imgset = "daal_tin"
# K = np.matrix("3368.26 0 1488.67; 0 3369.74 2023.21; 0 0 1")
# type_ = "jpg"

# n_imgs = 51  # Custom Dataset Blue Color Robot. 51 images taken from my mobile phone. first I did the camera calibration to find the K matrix
# imgset = "blue_boot_a53_4624x3468"
# K = np.matrix("2.25370759e+03 0.00000000e+00 1.92969309e+03; 0.00000000e+00 2.24471892e+03 1.05763445e+03; 0.00 0.00 1.00")
# type_ = "jpg"

# --- Create Output Directories ---
output_plots_base_dir = os.path.join(base_path, "output_plots", imgset)
features_out_dir = os.path.join(output_plots_base_dir, "features")
matches_out_dir = os.path.join(output_plots_base_dir, "feature_matches")

os.makedirs(features_out_dir, exist_ok=True)
os.makedirs(matches_out_dir, exist_ok=True)
print(f"Saving feature plots to: {features_out_dir}")
print(f"Saving match plots to: {matches_out_dir}")

images = get_images(base_path, imgset, type_, n_imgs, "gray")
images_color_for_plotting = get_images(base_path, imgset, type_, n_imgs)
assert len(images) == n_imgs
print(f"\n======== Using total {len(images)} images of dataset {imgset} ========\n\n\n")

feam_pipeline = SIFTMatcher()
keypoints, descriptors = feam_pipeline.extract_features(images)
print(f"\n======== Plotting and Saving Features (first few images as example) ========")
for i in range(n_imgs):
    img_for_plot = images_color_for_plotting[i] if images_color_for_plotting else images[i]
    save_plotted_keypoints(
        img_for_plot,
        keypoints[i],
        features_out_dir,
        f"features_img_{i:03d}.png",
        title=f"Detected {len(keypoints[i])} SIFT Features in Image {i}",
        show_plot=SHOW_PLOTS_INTERACTIVELY
    )
    
raw = feam_pipeline.match_all_pairs(descriptors)
matches = feam_pipeline.filter_outliers(raw, keypoints)
print("Matches:", feam_pipeline.count_total_matches(matches))
img_adjacency, list_of_img_pairs  = feam_pipeline.connectivity(matches)


print(f"\n======== Plotting and Saving Filtered Matches (example pairs) ========")
# Plot for a few example pairs that have matches
num_matches_to_plot = 0
max_match_plots =  n_imgs # Limit the number of match plots

for i in range(n_imgs):
    for j in range(i + 1, n_imgs):
        if matches[i][j] and len(matches[i][j]) > 0:
            img1_for_plot = images_color_for_plotting[i] if images_color_for_plotting else images[i]
            img2_for_plot = images_color_for_plotting[j] if images_color_for_plotting else images[j]
            
            save_plotted_matches(
                img1_for_plot, keypoints[i],
                img2_for_plot, keypoints[j],
                matches[i][j], # Draw the good (filtered) matches
                matches_out_dir,
                f"matches_{i:03d}_vs_{j:03d}.png",
                title=f"Filtered Matches {len(matches[i][j])} between Image {i} and Image {j}",
                show_plot=SHOW_PLOTS_INTERACTIVELY
            )
            num_matches_to_plot += 1

### This cell initializes the reconstruction
rec_pipeline = ReconstructionPipeline(img_adjacency, matches, keypoints, K)
best_pair = rec_pipeline.best_img_pair(top_x_perc=0.2)
R0, t0, R1, t1, points3d_with_views = rec_pipeline.initialize_reconstruction(best_pair[0], best_pair[1])

R_mats = {best_pair[0]: R0, best_pair[1]: R1}
t_vecs = {best_pair[0]: t0, best_pair[1]: t1}

resected_imgs = [best_pair[0], best_pair[1]] 
unresected_imgs = [i for i in range(len(images)) if i not in resected_imgs] 
print('initial image pair:', resected_imgs)
avg_err = 0


### This cell grows and refines the reconstruction 
BA_chkpts = [3,4,5,6] + [int(6*(1.34**i)) for i in range(25)]
while len(unresected_imgs) > 0:
    resected_idx, unresected_idx, prepend = rec_pipeline.next_img_pair_to_grow_reconstruction(n_imgs, best_pair, resected_imgs, unresected_imgs, img_adjacency)
    points3d_with_views, pts3d_for_pnp, pts2d_for_pnp, triangulation_status = rec_pipeline.get_correspondences_for_pnp(resected_idx, unresected_idx, points3d_with_views, matches, keypoints)
    if len(pts3d_for_pnp) < 12:
        print(f"{len(pts3d_for_pnp)} is too few correspondences for pnp. Skipping imgs resected:{resected_idx} and unresected:{unresected_idx}")
        print(f"Currently resected imgs: {resected_imgs}, unresected: {unresected_imgs}")
        continue

    R_res = R_mats[resected_idx]
    t_res = t_vecs[resected_idx]
    print(f"Unresected image: {unresected_idx}, resected: {resected_idx}")
    R_new, t_new = rec_pipeline.do_pnp(pts3d_for_pnp, pts2d_for_pnp, K)
    R_mats[unresected_idx] = R_new
    t_vecs[unresected_idx] = t_new
    if prepend == True: resected_imgs.insert(0, unresected_idx)
    else: resected_imgs.append(unresected_idx)
    unresected_imgs.remove(unresected_idx)
    pnp_errors, projpts, avg_err, perc_inliers = rec_pipeline.test_reproj_pnp_points(pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K)
    print(f"Average error of reprojecting points used to resect image {unresected_idx} back onto it is: {avg_err}")
    print(f"Fraction of Pnp inliers: {perc_inliers} num pts used in Pnp: {len(pnp_errors)}")
    
    if resected_idx < unresected_idx:
        kpts1, kpts2, kpts1_idxs, kpts2_idxs = rec_pipeline.get_aligned_kpts(resected_idx, unresected_idx, keypoints, matches, mask=triangulation_status)
        if np.sum(triangulation_status) > 0: #at least 1 point needs to be triangulated
            points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = rec_pipeline.triangulate_points_and_reproject(R_res, t_res, R_new, t_new, K, points3d_with_views, resected_idx, unresected_idx, kpts1, kpts2, kpts1_idxs, kpts2_idxs, compute_reproj=True)
    else:
        kpts1, kpts2, kpts1_idxs, kpts2_idxs = rec_pipeline.get_aligned_kpts(unresected_idx, resected_idx, keypoints, matches, mask=triangulation_status)
        if np.sum(triangulation_status) > 0: #at least 1 point needs to be triangulated
            points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = rec_pipeline.triangulate_points_and_reproject(R_new, t_new, R_res, t_res, K, points3d_with_views, unresected_idx, resected_idx, kpts1, kpts2, kpts1_idxs, kpts2_idxs, compute_reproj=True)
    
    if 0.8 < perc_inliers < 0.95 or 5 < avg_tri_err_l < 10 or 5 < avg_tri_err_r < 10: 
        #If % of inlers from Pnp is too low or triangulation error on either image is too high, bundle adjust
        if not USE_PYTORCH_OPTIMIZER:
            points3d_with_views, R_mats, t_vecs = b.do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol=1e0)
        else:
            points3d_with_views, R_mats, t_vecs = b.do_BA_pytorch(
                points3d_with_views, R_mats, t_vecs, 
                resected_imgs, # This is the list of original image indices active in BA
                keypoints, # This is your 'all_keypoints' list
                K, # Your camera matrix
                n_iterations=700, # Adjust as needed
                learning_rate=1e-45 # Adjust as needed
            )        
    if len(resected_imgs) in BA_chkpts or len(unresected_imgs) == 0 or perc_inliers <= 0.8 or avg_tri_err_l >= 10 or avg_tri_err_r >= 10:
        #If % of inlers from Pnp is very low or triangulation error on either image is very high, bundle adjust with stricter tolerance
        if not USE_PYTORCH_OPTIMIZER:
            points3d_with_views, R_mats, t_vecs = b.do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K, ftol=1e-1)
        else:
            points3d_with_views, R_mats, t_vecs = b.do_BA_pytorch(
                points3d_with_views, R_mats, t_vecs, 
                resected_imgs, # This is the list of original image indices active in BA
                keypoints, # This is your 'all_keypoints' list
                K, # Your camera matrix
                n_iterations=700, # Adjust as needed
                learning_rate=1e-45 # Adjust as needed
            )
        
    av = 0
    for im in resected_imgs:
        p3d, p2d, avg_error, errors = rec_pipeline.get_reproj_errors(im, points3d_with_views, R_mats[im], t_vecs[im], K, keypoints, distCoeffs=np.array([]))
        print(f'Average reprojection error on image {im} is {avg_error} pixels')
        av += avg_error
    av = av/len(resected_imgs)
    print(f'Average reprojection error across all {len(resected_imgs)} resected images is {av} pixels')
    
    
    
### This cell visualizes the pointcloud
num_voxels = 200 #Set to 100 for faster visualization, 200 for higher resolution.
x, y, z = [], [], []
for pt3 in points3d_with_views:
    if abs(pt3.point3d[0][0]) + abs(pt3.point3d[0][1]) + abs(pt3.point3d[0][2]) < 100:
        x.append(pt3.point3d[0][0])
        y.append(pt3.point3d[0][1])
        z.append(pt3.point3d[0][2])
vpoints = list(zip(x,y,z))
vpoints = np.array(vpoints)
vpoints_df = pd.DataFrame(data=vpoints, index=[f"{i}" for i in range(vpoints.shape[0])], columns=["x", "y","z"])
print("\nReconstruction finished. Exporting to COLMAP format...")

if images:
    img_h, img_w = images[0].shape[:2] # Assuming all images are same size for one camera model
else:
    print("No images loaded, cannot determine image dimensions for COLMAP export.")

# Define where to save COLMAP files
colmap_output_dir = os.path.join(base_path, "colmap_export", imgset) # example path

# Get image paths again or pass from where `get_images` was called
images_paths_for_export = sorted(
    glob.glob(
        os.path.join(base_path, "datasets", imgset) + "/*." + "png", # Assuming "png" or your img_format
        recursive=True,
    )
)

images_color_data = get_images(base_path, imgset, type_, n_imgs)

# Choose your color strategy: "first", "average", or "median"
chosen_color_strategy = "average" # Or "first" or "median"

export_to_colmap(
    output_path=colmap_output_dir,
    K_matrix=K,
    image_paths=images_paths_for_export,
    loaded_images=images_color_data, 
    all_keypoints=keypoints,
    reconstructed_R_mats=R_mats,
    reconstructed_t_vecs=t_vecs,
    reconstructed_points3d_with_views=points3d_with_views,
    image_height=img_h,
    image_width=img_w,
    point_color_strategy=chosen_color_strategy
)

visualize_sfm_open3d(vpoints)