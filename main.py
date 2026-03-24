import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

from modules.feature_extraction import FeatureExtractor
from modules.matching import FeatureMatcher
from modules.pose_estimation import PoseEstimator
from modules.utils import VOPlotter, parse_gt_pose


def main():
    # ══════════════════════════════════════════════════════════
    # 0. 路径配置
    # ══════════════════════════════════════════════════════════
    gt_path            = "data/dataset/poses/00.txt"
    image_dir          = "data/dataset/sequences/00/image_0/"
    save_plot_path     = "results/trajectory_00.png"
    save_matching_path = "results/matching_demo.png"
    os.makedirs("results", exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # 1. 初始化模块
    # ══════════════════════════════════════════════════════════
    extractor      = FeatureExtractor(num_features=2000, use_clahe=True)
    matcher        = FeatureMatcher(ratio_thresh=0.70)
    pose_estimator = PoseEstimator(inlier_ratio_thresh=0.50, min_matches=20, smooth_window=5)

    T_global = np.eye(4)

    # 检查 GT 文件
    if not os.path.exists(gt_path):
        print(f"❌ 找不到 Ground Truth 文件: {gt_path}")
        return
    plotter = VOPlotter(gt_path)

    # 检查图片目录
    if not os.path.exists(image_dir):
        print(f"❌ 图片路径不存在: {image_dir}")
        return
    images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith('.png')
    ])
    num_frames = min(len(images), plotter.num_frames)
    print(f"✅ 找到 {num_frames} 帧，开始处理...")

    # ══════════════════════════════════════════════════════════
    # 2. VO 主循环
    # ══════════════════════════════════════════════════════════
    for i in range(num_frames - 1):
        img1 = cv2.imread(images[i],   cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(images[i+1], cv2.IMREAD_GRAYSCALE)

        # ── 特征提取与匹配 ────────────────────────────────────
        kp1, des1 = extractor.extract_features(img1)
        kp2, des2 = extractor.extract_features(img2)
        matches    = matcher.match_features(des1, des2)

        # ── 位姿估计 ──────────────────────────────────────────
        R_rel, t_rel = pose_estimator.estimate_pose(kp1, kp2, matches)

        # ── 尺度校正（使用 GT 帧间距离）──────────────────────
        T_gt1   = parse_gt_pose(plotter.gt_poses_raw[i])
        T_gt2   = parse_gt_pose(plotter.gt_poses_raw[i+1])
        dist_gt = np.linalg.norm(T_gt1[:3, 3] - T_gt2[:3, 3])

        # 额外保护：GT 帧间距离极小时（车辆几乎静止），跳过累加
        if dist_gt < 1e-3:
            plotter.add_estimated_pose(T_global)
            continue

        t_scaled = t_rel * dist_gt

        # ── 位姿累加 ──────────────────────────────────────────
        T_rel_scaled = np.eye(4)
        T_rel_scaled[:3, :3] = R_rel
        T_rel_scaled[:3, 3]  = t_scaled.flatten()

        T_global = T_global @ T_rel_scaled
        plotter.add_estimated_pose(T_global)

        # ── 进度 & 保存第一帧匹配图 ───────────────────────────
        if i % 100 == 0:
            print(f"  Frame {i:4d}/{num_frames}  matches={len(matches):4d}")
            if i == 0 and len(matches) > 0:
                matching_img = matcher.draw_matches(img1, kp1, img2, kp2, matches)
                cv2.imwrite(save_matching_path, matching_img)

    # ══════════════════════════════════════════════════════════
    # 3. 输出结果
    # ══════════════════════════════════════════════════════════
    plotter.save_trajectory_plot(save_plot_path)
    print("🏁 处理完成，请查看 results/ 目录。")


if __name__ == "__main__":
    main()