import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        # KITTI 00 序列内参矩阵 K
        self.K = np.array([[718.856, 0.0, 607.1928],
                          [0.0, 718.856, 185.2157],
                          [0.0, 0.0, 1.0]])

    def estimate_pose(self, kp1, kp2, matches):
        # 提取匹配点的 2D 坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # 计算本质矩阵 E
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=0.5)

        # 恢复位姿
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # 🔴 核心修正：求逆！
        # OpenCV 返回的是从 1 到 2 的变换，我们需要 2 在 1 坐标系下的位姿
        R_rel = R.T
        t_rel = -R.T @ t

        return R_rel, t_rel