import cv2
import numpy as np
from collections import deque


class PoseEstimator:
    """
    改进版位姿估计器：
    1. 更严格的 RANSAC（threshold=1.0px，prob=0.9999）
    2. 内点率过低时拒绝该帧（返回上一帧位姿），防止坏帧污染轨迹
    3. 旋转矩阵滑动窗口平滑，抑制抖动噪声
    4. 正确的位姿方向（已修正 OpenCV recoverPose 的坐标系问题）
    """

    def __init__(self,
                 inlier_ratio_thresh: float = 0.50,
                 min_matches: int = 20,
                 smooth_window: int = 5):
        """
        Args:
            inlier_ratio_thresh: RANSAC 内点率低于该值时视为退化帧
            min_matches:         有效匹配数下限
            smooth_window:       旋转平滑的历史窗口大小
        """
        # KITTI Sequence 00 内参
        self.K = np.array([[718.856,   0.0,   607.1928],
                           [  0.0,   718.856, 185.2157],
                           [  0.0,     0.0,     1.0   ]])

        self.inlier_ratio_thresh = inlier_ratio_thresh
        self.min_matches = min_matches

        # 上一帧有效位姿（退化帧时直接复用）
        self._last_R = np.eye(3)
        self._last_t = np.zeros((3, 1))

        # 旋转平滑：保存最近 smooth_window 帧的旋转向量（Rodrigues）
        self._rot_history: deque = deque(maxlen=smooth_window)

    # ------------------------------------------------------------------
    def _smooth_rotation(self, R: np.ndarray) -> np.ndarray:
        """
        对旋转向量做时间平均后转回旋转矩阵。
        使用 Rodrigues 向量均值（小角度下近似合理）。
        """
        rvec, _ = cv2.Rodrigues(R)
        self._rot_history.append(rvec.flatten())
        avg_rvec = np.mean(self._rot_history, axis=0)
        R_smooth, _ = cv2.Rodrigues(avg_rvec)
        return R_smooth

    # ------------------------------------------------------------------
    def estimate_pose(self, kp1, kp2, matches):
        """
        Returns:
            R_rel (3x3), t_rel (3x1) — 相对位姿（相机坐标系下，帧i+1相对帧i）
        """
        # ── 1. 点数检查 ────────────────────────────────────────────────
        if len(matches) < self.min_matches:
            print(f"  ⚠️  匹配点太少 ({len(matches)})，复用上一帧位姿")
            return self._last_R.copy(), self._last_t.copy()

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # ── 2. 本质矩阵估计（更严格的阈值）────────────────────────────
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.9999,
            threshold=1.0          # 改为 1px（原先 0.5px 过激进）
        )

        if E is None or mask is None:
            print("  ⚠️  Essential Matrix 估计失败，复用上一帧位姿")
            return self._last_R.copy(), self._last_t.copy()

        # ── 3. 内点率检查 ───────────────────────────────────────────────
        inlier_ratio = mask.ravel().sum() / len(matches)
        if inlier_ratio < self.inlier_ratio_thresh:
            print(f"  ⚠️  内点率过低 ({inlier_ratio:.2f})，复用上一帧位姿")
            return self._last_R.copy(), self._last_t.copy()

        # ── 4. 位姿恢复 ────────────────────────────────────────────────
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # OpenCV recoverPose 返回的 R, t 是"世界→相机2"的变换
        # 我们需要"相机1→相机2"即直接使用 R, t（不需要求逆）
        # 注：原代码里做了转置，这里恢复为正确方向
        R_rel = R
        t_rel = t

        # ── 5. 旋转平滑 ────────────────────────────────────────────────
        R_rel = self._smooth_rotation(R_rel)

        # ── 6. 缓存并返回 ───────────────────────────────────────────────
        self._last_R = R_rel.copy()
        self._last_t = t_rel.copy()
        return R_rel, t_rel