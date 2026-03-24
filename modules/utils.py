import numpy as np
import matplotlib.pyplot as plt

class VOPlotter:
    def __init__(self, gt_path):
        # 1. 加载 KITTI Ground Truth 轨迹
        # KITTI Ground Truth 格式是 N 行 12 列，每一行是展开的 3x4 变换矩阵 (R|t)
        self.gt_poses = np.loadtxt(gt_path)
        self.num_frames = len(self.gt_poses)
        
        # 提取 GT 轨迹的 (x, z) 坐标用于绘图 (KITTI 中相机运动主平面是 X-Z 平面)
        self.gt_x = self.gt_poses[:, 3]
        self.gt_z = self.gt_poses[:, 11]

        # 2. 初始化用于存储估计轨迹的列表
        self.est_x = [0.0]
        self.est_z = [0.0]

    def add_estimated_pose(self, T_global):
        """
        接收当前的全局位姿变换矩阵 (4x4)，提取它的平移部分 (x, z)
        """
        self.est_x.append(T_global[0, 3])
        self.est_z.append(T_global[2, 3])

    def save_trajectory_plot(self, save_path):
        """
        绘制轨迹对比图并保存到 results/ 目录
        """
        plt.figure(figsize=(10, 8))
        
        # 绘制 Ground Truth (蓝线)
        plt.plot(self.gt_x, self.gt_z, label='Ground Truth', color='blue', linewidth=1)
        
        # 绘制估计轨迹 (红虚线)
        # 注意：因为帧数不匹配问题（估计少了一帧），我们需要对 est 列表切片
        plt.plot(self.est_x, self.est_z, label='Estimated VO (Scale Corrected)', 
                 color='red', linestyle='--', linewidth=2)
        
        plt.title('Visual Odometry on KITTI Sequence 00', fontsize=15)
        plt.xlabel('X (meters)', fontsize=12)
        plt.ylabel('Z (meters)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # 保存图片
        plt.savefig(save_path)
        plt.close()
        print(f"✅ 轨迹图已成功保存至: {save_path}")

# 一个辅助函数，用于将 KITTI 数据集的 N*12 格式转换为 4x4 矩阵
def parse_gt_pose(gt_frame):
    """
    输入一行 12 个数，输出 4x4 变换矩阵
    """
    T = np.eye(4)
    T[:3, :4] = gt_frame.reshape(3, 4)
    return T