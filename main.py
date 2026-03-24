import cv2
import os
import numpy as np
import matplotlib
# Use 'Agg' backend to ensure plots are saved to file without popping up a window
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Import custom modules
from modules.feature_extraction import FeatureExtractor
from modules.matching import FeatureMatcher
from modules.pose_estimation import PoseEstimator
from modules.utils import VOPlotter, parse_gt_pose

def main():
    # ==============================
    # 0. Path Configuration
    # ==============================
    # Ensure poses.txt is present (often a separate download from the main KITTI grayscale pack)
    gt_path = "data/dataset/poses/00.txt" 
    image_dir = "data/dataset/sequences/00/image_0/"
    save_plot_path = "results/trajectory_00.png"
    save_matching_path = "results/matching_demo.png" # For README visualization
    save_matching_path = "results/matching_demo.gif"
    # ==============================
    # 1. Initialize Modules
    # ==============================
    extractor = FeatureExtractor()
    matcher = FeatureMatcher()
    pose_estimator = PoseEstimator() 
    
    # Initialize global transformation matrix T_global as Identity (Starting pose)
    T_global = np.eye(4)
    
    # Initialize plotting tools
    if not os.path.exists(gt_path):
        print(f"❌ Ground Truth file not found: {gt_path}. Scale correction and comparison cannot be performed.")
        return
    plotter = VOPlotter(gt_path)

    # Path verification
    if not os.path.exists(image_dir):
        print(f"❌ Image directory does not exist: {image_dir}")
        return
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    
    # Match frame counts between images and GT
    num_frames = min(len(images), plotter.num_frames)
    print(f"✅ Found {num_frames} images and corresponding Ground Truth. Starting processing...")

    matching_frames = []
    capture_start_frame = 100
    max_to_capture = 60

    # ==============================
    # 2. VO Loop Processing
    # ==============================
    for i in range(num_frames - 1):
        # Read adjacent frames
        img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(images[i+1], cv2.IMREAD_GRAYSCALE)

        # Feature extraction and matching
        kp1, des1 = extractor.extract_features(img1)
        kp2, des2 = extractor.extract_features(img2)
        matches = matcher.match_features(des1, des2)

        # Pose estimation (Relative rotation R and translation t with unknown scale)
        R_rel, t_rel = pose_estimator.estimate_pose(kp1, kp2, matches)
        if R_rel is None:
            continue  # skip degenerate frame
        
        # ==========================================
        # 3. Core: Scale Correction (using Ground Truth)
        # ==========================================
        # Essential for Monocular VO to prevent trajectory drift and incorrect scale.
        
        # Get GT poses to calculate true scale
        T_gt1 = parse_gt_pose(plotter.gt_poses[i])
        T_gt2 = parse_gt_pose(plotter.gt_poses[i+1])
        
        # Calculate Ground Truth travel distance (Scale)
        # Using Euclidean distance between translation vectors of two frames
        dist_gt = np.linalg.norm(T_gt1[:3, 3] - T_gt2[:3, 3])
        
        # Scale the estimated translation vector by the GT scale
        t_scaled = t_rel * dist_gt
        
        # ==========================================
        # 4. Trajectory Integration
        # ==========================================
        # T_global_{i+1} = T_global_{i} * T_rel_{scaled}
        
        # Construct current relative transformation matrix (4x4)
        T_rel_scaled = np.eye(4)
        T_rel_scaled[:3, :3] = R_rel
        T_rel_scaled[:3, 3] = t_scaled.flatten()
        
        # Accumulate: update global pose
        T_global = T_global @ T_rel_scaled 

        # Add global pose to plotter
        plotter.add_estimated_pose(T_global)

        # Print progress
        if i % 100 == 0:
            print(f"Processing frame {i}/{num_frames}...")
            
        if i >= capture_start_frame and len(matching_frames) < max_to_capture:
            matching_img = matcher.draw_matches(img1, kp1, img2, kp2, matches)
            matching_frames.append(matching_img)

    # ==============================
    # 5. Generate Results
    # ==============================
    plotter.save_trajectory_plot(save_plot_path)
    print("🏁 Processing finished. Please check the results/ directory for images.")
    plotter.save_trajectory_gif("results/trajectory_00.gif")

    if matching_frames:
        matcher.save_matching_gif(matching_frames, save_matching_path)

if __name__ == "__main__":
    main()
