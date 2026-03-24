import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────────────

def parse_gt_pose(gt_frame: np.ndarray) -> np.ndarray:
    """将 KITTI 的一行 12 个数转为 4×4 变换矩阵"""
    T = np.eye(4)
    T[:3, :4] = gt_frame.reshape(3, 4)
    return T


# ──────────────────────────────────────────────────────────────────────────────
# VOPlotter
# ──────────────────────────────────────────────────────────────────────────────

class VOPlotter:
    """
    改进版画图 & 评估工具：
    - 对齐起点（两条轨迹都从 (0,0) 出发）
    - 计算 ATE（绝对轨迹误差）和 RTE（相对轨迹误差）
    - 可选保存误差曲线图
    """

    def __init__(self, gt_path: str):
        self.gt_poses_raw = np.loadtxt(gt_path)
        self.num_frames = len(self.gt_poses_raw)

        # GT 轨迹坐标（以第一帧为原点）
        x0 = self.gt_poses_raw[0, 3]
        z0 = self.gt_poses_raw[0, 11]
        self.gt_x = self.gt_poses_raw[:, 3] - x0
        self.gt_z = self.gt_poses_raw[:, 11] - z0

        # 缓存 GT 4×4 矩阵
        self._gt_T = [parse_gt_pose(row) for row in self.gt_poses_raw]

        # 估计轨迹
        self.est_x = [0.0]
        self.est_z = [0.0]

    # ── 写入估计位姿 ──────────────────────────────────────────────────────────
    def add_estimated_pose(self, T_global: np.ndarray):
        self.est_x.append(float(T_global[0, 3]))
        self.est_z.append(float(T_global[2, 3]))

    # ── 误差计算 ──────────────────────────────────────────────────────────────
    def compute_ate(self) -> float:
        """
        绝对轨迹误差（ATE, RMSE）：
        每帧估计位置与 GT 位置的欧氏距离的均方根
        """
        n = min(len(self.est_x), len(self.gt_x))
        ex = np.array(self.est_x[:n]) - self.gt_x[:n]
        ez = np.array(self.est_z[:n]) - self.gt_z[:n]
        ate = float(np.sqrt(np.mean(ex**2 + ez**2)))
        return ate

    def compute_rte(self, segment_len: int = 100) -> float:
        """
        相对轨迹误差（RTE）：
        每隔 segment_len 帧计算一次相对平移误差，取平均
        """
        n = min(len(self.est_x), len(self.gt_x))
        errors = []
        for i in range(0, n - segment_len, segment_len):
            j = i + segment_len
            # 估计的位移增量
            de_x = self.est_x[j] - self.est_x[i]
            de_z = self.est_z[j] - self.est_z[i]
            # GT 的位移增量
            dg_x = self.gt_x[j] - self.gt_x[i]
            dg_z = self.gt_z[j] - self.gt_z[i]
            err = np.sqrt((de_x - dg_x)**2 + (de_z - dg_z)**2)
            errors.append(err)
        return float(np.mean(errors)) if errors else float('nan')

    # ── 轨迹图 ────────────────────────────────────────────────────────────────
    def save_trajectory_plot(self, save_path: str):
        ate = self.compute_ate()
        rte = self.compute_rte()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ── 左图：轨迹对比 ─────────────────────────────────────────────
        ax = axes[0]
        ax.plot(self.gt_x, self.gt_z,
                label='Ground Truth', color='royalblue', linewidth=1.5)
        ax.plot(self.est_x, self.est_z,
                label=f'Estimated VO\nATE={ate:.2f}m  RTE={rte:.2f}m',
                color='crimson', linestyle='--', linewidth=1.5)
        ax.scatter([0], [0], c='green', s=80, zorder=5, label='Start')
        ax.set_title('Trajectory Comparison (KITTI Seq 00)', fontsize=13)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        ax.set_aspect('equal')

        # ── 右图：逐帧误差曲线 ─────────────────────────────────────────
        ax2 = axes[1]
        n = min(len(self.est_x), len(self.gt_x))
        ex = np.array(self.est_x[:n]) - self.gt_x[:n]
        ez = np.array(self.est_z[:n]) - self.gt_z[:n]
        per_frame_err = np.sqrt(ex**2 + ez**2)
        ax2.plot(per_frame_err, color='darkorange', linewidth=1)
        ax2.axhline(ate, color='red', linestyle='--', label=f'ATE={ate:.2f}m')
        ax2.set_title('Per-frame Position Error', fontsize=13)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Error (meters)')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ 轨迹图保存至: {save_path}")
        print(f"   📊 ATE (RMSE) = {ate:.3f} m")
        print(f"   📊 RTE (100-frame avg) = {rte:.3f} m")