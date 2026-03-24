import cv2
import os
import numpy as np
# 引入matplotlib用于画图，这在服务器环境可能需要切换后端
import matplotlib
matplotlib.use('Agg') # 确保不弹出窗口，直接后台画图保存
import matplotlib.pyplot as plt

# 引入自定义模块
from modules.feature_extraction import FeatureExtractor
from modules.matching import FeatureMatcher
from modules.pose_estimation import PoseEstimator
from modules.utils import VOPlotter, parse_gt_pose # 新增

def main():
    # ==============================
    # 0. 路径配置 (核心！)
    # ==============================
    # 请确保你有这个 poses.txt 文件，它不在 grayscale 22GB 压缩包里，可能需要单独下载 KITTI "poses" 文件
    # 如果实在没有，把 gt_path 设为 None，我会屏蔽掉 GT 绘图功能
    gt_path = "data/dataset/poses/00.txt" 
    image_dir = "data/dataset/sequences/00/image_0/"
    save_plot_path = "results/trajectory_00.png"
    save_matching_path = "results/matching_demo.png" # 用于保存第一帧匹配图放README

    # ==============================
    # 1. 初始化所有模块
    # ==============================
    extractor = FeatureExtractor()
    matcher = FeatureMatcher()
    pose_estimator = PoseEstimator() # 确保已经在 pose_estimation.py 里写好了 K
    
    # 初始化全局变换矩阵 T_global 为单位矩阵 (相机起始位姿)
    T_global = np.eye(4)
    
    # 初始化画图工具
    if not os.path.exists(gt_path):
        print(f"❌ 找不到 Ground Truth 文件: {gt_path}。我们将无法进行尺度校正和画对比图。")
        return
    plotter = VOPlotter(gt_path)

    # 路径确认
    if not os.path.exists(image_dir):
        print(f"❌ 图片路径不存在: {image_dir}")
        return
    images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    
    # 确保 data/sequences_num.txt 加载的帧数与图片数匹配
    num_frames = min(len(images), plotter.num_frames)
    print(f"✅ 找到 {num_frames} 张图片和对应Ground Truth，准备开始处理...")

    # ==============================
    # 2. VO 循环处理
    # ==============================
    for i in range(num_frames - 1):
        # 读取相邻两帧
        img1 = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(images[i+1], cv2.IMREAD_GRAYSCALE)

        # 特征提取与匹配
        kp1, des1 = extractor.extract_features(img1)
        kp2, des2 = extractor.extract_features(img2)
        matches = matcher.match_features(des1, des2)

        # 位姿估计 (得到相对旋转 R 和 尺度待定的平移 t)
        R_rel, t_rel = pose_estimator.estimate_pose(kp1, kp2, matches)
        
        # ==========================================
        # 3. 核心：尺度校正 (用Ground Truth辅助)
        # ==========================================
        # 这一步对于单目 VO 必不可少，否则由于尺度漂移，轨迹完全不可看。
        
        # 读取 GT 位姿用于计算真实尺度
        T_gt1 = parse_gt_pose(plotter.gt_poses[i])
        T_gt2 = parse_gt_pose(plotter.gt_poses[i+1])
        
        # 计算 Ground Truth 的移动距离 (尺度)
        # 即两帧之间平移向量的欧氏距离
        dist_gt = np.linalg.norm(T_gt1[:3, 3] - T_gt2[:3, 3])
        
        # 将估计的平移向量乘以 GT 的尺度
        t_scaled = t_rel * dist_gt
        
        # ==========================================
        # 4. 位姿累加 (Trajectory Integration)
        # ==========================================
        # T_global_{i+1} = T_global_{i} * T_rel_{scaled}
        
        # 构造当前的相对变换矩阵 (4x4)
        T_rel_scaled = np.eye(4)
        T_rel_scaled[:3, :3] = R_rel
        T_rel_scaled[:3, 3] = t_scaled.flatten()
        
        # 累加：更新全局位姿
        T_global = T_global @ T_rel_scaled # Python 3.5+ 矩阵乘法简写

        # 将累加后的全局位姿交给画图工具
        plotter.add_estimated_pose(T_global)

        # 打印进度
        if i % 100 == 0:
            print(f"Processing frame {i}/{num_frames}...")
            # 保存第1帧的匹配图放README
            if i == 0:
                matching_img = matcher.draw_matches(img1, kp1, img2, kp2, matches)
                cv2.imwrite(save_matching_path, matching_img)

    # ==============================
    # 5. 生成结果
    # ==============================
    # 循环结束后，画图并保存
    plotter.save_trajectory_plot(save_plot_path)
    print("🏁 全部处理结束，请查看 results/ 目录下的图片。")

if __name__ == "__main__":
    main()